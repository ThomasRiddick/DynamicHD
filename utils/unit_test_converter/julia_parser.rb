class JuliaUnitTest
  attr_reader :name, :body, :statements
  attr_writer :name, :body, :statements
  def initialize(name,body)
    @name=name
    @body=body
    @statements = Array.new
  end
end

class JuliaStatement
  attr_reader :type,:content
  attr_writer :type,:content
  def initialize(type,content)
    @type = type
    @content = content
  end
end

class JuliaParser
  @@test_pattern=/^\s*@testset\s*"(\s*[\w\s\d]+)\s*"\s*begin(\s+|$)/
  @@open_block_pattern=/^s\*begin(\s+|$)/
  @@if_pattern=/^\s*if\s+/
  @@single_line_if_pattern=/^\s*if\s*.*\s*end(\s+|$)/
  @@close_block_pattern=/(?:^\s*|\s+)end(\s+|$)/
  @@julia_loop_construct=/^\s*for\s+(\S+)\s+in\s+(\S+)\s*$/
  @@julia_numeric_loop_construct=/^\s*for\s*(\S+)\s*=\s*(\d+)\s*:\s*(\d+)\s*$/
  @@julia_open_construct=Regexp.union(@@test_pattern,@@open_block_pattern,
                                      @@single_line_if_pattern,@@if_pattern,
                                      @@julia_loop_construct,@@julia_numeric_loop_construct)
  @@julia_comment_line=/\s*#.*$/
  @@julia_double_field_definition=
    /\s*(\w+)(?:::(?:LatLon)?Field\{Float64\})?\s*=\s*LatLonField\{\s*Float64\s*\}\(.*,(.*)\).*$/
  @@julia_int_field_definition=
    /\s*(\w+)(?:::(?:LatLon)?Field\{Int64\})?\s*=\s*LatLonField\{\s*Int64\s*\}\(.*,(.*)\).*$/
  @@julia_bool_field_definition=
    /\s*(\w+)(?:::(?:LatLon)?Field\{Bool\})?\s*=\s*LatLonField\{\s*Bool\s*\}\(.*,(.*)\).*$/
  @@julia_initialised_double_field_definition=
    /\s*(\w+)(?:::(?:LatLon)?Field\{Float64\})?\s*=\s*LatLonField\{\s*Float64\s*\}\([^)&]*($|&)/
  @@julia_initialised_int_field_definition=
    /\s*(\w+)(?:::(?:LatLon)?Field\{Int64\})?\s*=\s*LatLonField\{\s*Int64\s*\}\([^)&]*($|&)/
  @@julia_initialised_bool_field_definition=
    /\s*(\w+)(?:::(?:LatLon)?Field\{Bool\})?\s*=\s*LatLonField\{\s*Bool\s*\}\([^)&]*($|&)/
  @@julia_initialised_user_defined_type_field_definition=
    /\s*(\w+)(?:::(?:LatLon)?Field\{\s*(?!(Bool|Int64|Float64)).*\})?\s*=\s*LatLonField\{\s*(?!(Bool|Int64|Float64)).*\s*\}\([^)&]*($|&)/
  @@julia_double_definition=/\s*\w+::Float64\s*=\s*(\d+\.d*)/
  @@julia_int_definition=/\s*\w+::Int64\s*=\s*(\d+)/
  @@julia_bool_definition=/\s*\w+::Bool\s*=\s*(true|false)/
  @@julia_function_call=/^\s*\w+\(.*\)\s*$/
  @@multi_line_julia_function_call=/^\s*\w+\([^)]*$/
  @@julia_assignment=/^\s*\w+\s*=.*\).*$/
  @@multi_line_julia_assignment=/^\s*\w+\s*=[^)]*$/
  @@julia_array_assignment=/^\s*set!\((\w+)\s*,\s*LatLonCoords\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*(.*)\s*\)/
  @@julia_assertion=/^\s*@test\s+/
  attr_reader :tests
  def initialize(text)
    @text = text
    @tests = Array.new
  end

  def parse
    identify_test_definitions
    identify_statements
  end

  def identify_test_definitions
    while @text =~ @@test_pattern
      test_name=$1
      @text=$'
      @text=$'
      text_including_current_test = @text.clone
      @text = find_closing_statement(@text)
      current_test_length = text_including_current_test.length - @text.length
      @tests.push(JuliaUnitTest.new(test_name,
                                    text_including_current_test[0...current_test_length-1]))
    end
  end

  def find_closing_statement(text)
    open_blocks=1
    loop do
      if close_block_match = @@close_block_pattern.match(text)
        unless close_block_match.pre_match =~ @@julia_open_construct then
          text=close_block_match.post_match
          open_blocks -= 1
          return text if open_blocks == 0
          next
        end
      end
      if text =~ @@julia_open_construct
        text=$'
        if not ($& =~ @@single_line_if_pattern)
          open_blocks += 1
        end
      end
    end
  end

  def identify_statements
    for test in @tests
      statements = Array.new
      sanitized_text = test.body.clone
      sanitized_text.gsub!(/#=\s*\n*\s*=#/," ")
      sanitized_text.gsub!(/(.*)=\s*\n/,"\\1=")
      # This is hardwired to convert to Fortran. Need redevelopment to be more
      # abstract as Base class should be general
      # Have also removed loop for the end of the loop as it wasn't work
      # Also need to deal with if tests
      sanitized_text.gsub!(@@julia_loop_construct,"PROCESS BY HAND=>do\\1 in \\2 #\n")
      sanitized_text.gsub!(@@julia_numeric_loop_construct,
                                "PROCESS BY HAND=>do \\1 = \\2,\\3 #\n")
      sanitized_text.gsub!(@@julia_comment_line,'')
      content = ""
      while endline_match = sanitized_text.match(/\n/)
        sanitized_text=endline_match.post_match
        type =  case endline_match.pre_match.strip
                  when @@julia_initialised_double_field_definition then :double_initialised_array_definition
                  when @@julia_initialised_int_field_definition then :int_initialised_array_definition
                  when @@julia_initialised_bool_field_definition then :bool_initialised_array_definition
                  when @@julia_initialised_user_defined_type_field_definition \
                    then :user_defined_type_initialised_array_definition
                  when @@julia_double_field_definition then :double_array_definition
                  when @@julia_int_field_definition then :int_array_definition
                  when @@julia_bool_field_definition then :bool_array_definition
                  when @@julia_double_definition then :double_definition
                  when @@julia_int_definition then :int_definition
                  when @@julia_bool_definition then :bool_definition
                  when @@julia_assertion then :julia_assertion
                  when @@julia_function_call then :julia_function_call
                  when @@multi_line_julia_function_call then :multi_line_julia_function_call
                  when @@julia_assignment then :assignment
                  when @@multi_line_julia_assignment then :multi_line_assignment
                  when @@julia_array_assignment then :array_assignment
                  else :unknown_statement_type
                end
        contents = endline_match.pre_match.strip
        case type
          when :double_initialised_array_definition,:int_initialised_array_definition,
               :bool_initialised_array_definition, :multi_line_julia_function_call,
               :multi_line_assignment, :user_defined_type_initialised_array_definition
            contents += " &\n"
            while endline_match = sanitized_text.match(/\n/)
              sanitized_text=endline_match.post_match
              contents += "         " + endline_match.pre_match.strip
              break if endline_match.pre_match.match(/\)/)
              contents += " &\n"
            end
        end
        statements.push(JuliaStatement.new(type,contents))
      end
      test.statements = statements
    end
  end
end

class JuliaToFortranConverter < JuliaParser

  def initialize(text)
    super(text)
  end

  def convert_julia_tests_to_fortran
    fortran_unit_tests=""
    for test in @tests
      fortran_unit_tests+=convert_julia_test_to_fortran(test)
    end
    return fortran_unit_tests
  end

  def convert_julia_test_to_fortran(test)
    fortran_unit_test_name="subroutine test" +
                            test.name.sub(/ tests/,"").split.map{|s| s.capitalize}.join
    fortran_unit_test="\n" + fortran_unit_test_name
    fortran_statements=""
    fortran_definitions=""
    for statement in test.statements
      fortran_statements +="\n      "
      case statement.type
        when :double_initialised_array_definition
          fortran_definitions+=generate_array_definition(statement.content,type=:real)
          fortran_statements +=convert_initialised_array(statement.content,type=:real)
        when :int_initialised_array_definition
          fortran_definitions+=generate_array_definition(statement.content,type=:integer)
          fortran_statements +=convert_initialised_array(statement.content,type=:integer)
        when :bool_initialised_array_definition
          fortran_definitions+=generate_array_definition(statement.content,type=:logical)
          fortran_statements +=convert_initialised_array(statement.content,type=:logical)
        when :user_defined_type_initialised_array_definition
          fortran_definitions+=generate_array_definition(statement.content,type=:user_defined)
          fortran_statements +=convert_initialised_array(statement.content,type=:user_defined)
        when :double_array_definition
          fortran_definitions+=generate_array_definition(statement.content,type=:real)
          fortran_statements +=convert_uninitialised_array(statement.content,type=:real)
        when :int_array_definition
          fortran_definitions+=generate_array_definition(statement.content,type=:integer)
          fortran_statements +=convert_uninitialised_array(statement.content,type=:integer)
        when :bool_array_definition
          fortran_definitions+=generate_array_definition(statement.content,type=:logical)
          fortran_statements +=convert_uninitialised_array(statement.content,type=:logical)
        when :double_definition
          fortran_definitions+=generate_definition(statement.content,type=:real)
          fortran_statements +=statement.content.gsub(/::Float64/,"").squeeze(" ").strip
        when :int_definition
          fortran_definitions+=generate_definition(statement.content,type=:integer)
          fortran_statements +=statement.content.gsub(/::Int64/,"").squeeze(" ").strip
        when :bool_definition
          fortran_definitions+=generate_definition(statement.content,type=:logical)
          fortran_statements +=statement.content.gsub(/::Bool/,"").gsub(/true/,"True").
            gsub(/false/,"False").squeeze(" ").strip
        when :julia_assertion
          fortran_statements +="PROCESS BY HAND=>?call assert_equals("+statement.content.
            gsub(/true/,"True").gsub(/false/,"False").sub(/\s*@test/,"      (").strip + (")")
        when :julia_function_call
          fortran_statements +="PROCESS BY HAND=>call "+statement.content.gsub(/true/,".True.").gsub(/false/,".False.").strip
        when :multi_line_julia_function_call
          fortran_statements +="PROCESS BY HAND=>call "+statement.content.gsub(/true/,".True.").gsub(/false/,".False.").strip
        when :assignment
          fortran_statements +="PROCESS BY HAND=> "+statement.content.gsub(/true/,"True").gsub(/false/,"False").squeeze(" ").strip
        when :multi_line_assignment
          fortran_statements +="PROCESS BY HAND=>"+statement.content.gsub(/true/,"True").gsub(/false/,"False").
                                                        squeeze(" ").strip.gsub(/\n/,"\n         ")
        when :array_assignment
          fortran_statements +=statement.content.gsub(/true/,".True.").gsub(/false/,"False.").
                               squeeze(" ").strip.sub(@@julia_array_assignment,"\\1(\\2,\\3) = \\4")
        when :unknown_statement_type
          fortran_statements +="PROCESS BY HAND=>"+statement.content.strip
        else raise "Incorrectly labelled statement"
      end
    end
    fortran_unit_test += "\n" + fortran_definitions
    fortran_unit_test += fortran_statements
    fortran_unit_test += "\nend " + fortran_unit_test_name + "\n"
    return fortran_unit_test
  end

  def convert_initialised_array(array_statement,type)
    start_pattern = case type
                      when :real then @@julia_initialised_double_field_definition
                      when :integer then @@julia_initialised_int_field_definition
                      when :logical then @@julia_initialised_bool_field_definition
                      when :user_defined \
                        then @@julia_initialised_user_defined_type_field_definition
                    end
    case type
      when :real then array_statement.sub!("Float64[","")
      when :integer then array_statement.sub!("Int64[","")
      when :logical then array_statement.sub!("Bool[","")
    end
    array_statement.sub!(start_pattern,"allocate(\\1(x,y))\n      \\1 = transpose(reshape((/ &")
    array_statement.gsub!(/([\d.]+) /,"\\1, ")
    array_statement.gsub!(/([\w.]+) /,"\\1, ")
    array_statement.gsub!(/true,/,".True.,")
    array_statement.gsub!(/false,/,".False.,")
    array_statement.sub!(/\]\)/,"/), &\n         (/y,x/)))")
    if type == :real
      array_statement.gsub!(/ 0,/,"0.0,")
    end
    return array_statement
  end

  def convert_uninitialised_array(array_statement,type)
    pattern = case type
                  when :real then  @@julia_double_field_definition
                  when :integer then @@julia_int_field_definition
                  when :logical then @@julia_bool_field_definition
              end
    array_statement.sub!(pattern,"allocate(\\1(x,y))\n      \\1(:,:) = \\2")
    return array_statement
  end

  def generate_definition(array_statement,type)
    name = array_statement[/^\s*(\w+)(?:::(?:Bool|Int64|Float64))?\s*=.*/,1]
    return "   " + type.to_s + " :: #{name} \n"
  end

  def generate_array_definition(array_statement,type)
    name = array_statement[/^\s*(\w+)(?:::(?:LatLon)?Field\{[\w\d]*\})?\s*=.*/,1]
    return "   " + type.to_s + ",dimension(:,:), allocatable :: #{name} \n"
  end
end
