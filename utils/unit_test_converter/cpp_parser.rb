class CppUnitTest
  attr_reader :name, :body, :statements
  attr_writer :name, :body, :statements
  def initialize(name,body)
    @name=name
    @body=body
    @statements = Array.new
  end
end

class CppStatement
  attr_reader :type,:content
  attr_writer :type,:content
  def initialize(type,content)
    @type = type
    @content = content
  end
end

class CppParser
  @@test_pattern=/^ *TEST(_F)? */
  @@test_location_and_name_pattern=/ *\( *\w+ *, *(\w+) *\)/
  @@curly_open_bracket_pattern=/ *\{/
  @@curly_close_bracket_pattern=/ *\}/
  @@cpp_loop_construct=/ *for *\(.*;.*;.*\) *\{/
  @@cpp_comment_line=/ *\/\/.*$/
  @@cpp_double_array_definition=/\s*double\s*\*\s*\w+\s*=\s*new/
  @@cpp_int_array_definition=/\s*int\s*\*\s*\w+\s*=\s*new/
  @@cpp_bool_array_definition=/\s*bool\s*\*\s*\w+\s*=\s*new/
  @@cpp_double_definition=/\s*double\s*\w+\s*=\s*\d/
  @@cpp_int_definition=/\s*int\s*\w+\s*=\s*\d/
  @@cpp_bool_definition=/\s*bool\s*\w+\s*=\s*(true|false)/
  @@cpp_delete=/^\s*delete(\[\])?\s*/
  @@cpp_function_call=/^\s*\w+\(.*\)\s*$/m
  @@cpp_assignment=/^\s*\w+\s*=/
  @@cpp_array_assignment=/^\s*\w+\[.*\]\s*=/m
  @@gtest_assertion=/^\s*EXPECT_/
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
      @text=$'
      if @text =~ @@curly_open_bracket_pattern
        @text=$'
        if $` =~ @@test_location_and_name_pattern then test_name=$1
        end
        text_including_current_test = @text.clone
        @text = find_closing_curly_bracket(@text)
        current_test_length = text_including_current_test.length - @text.length
        @tests.push(CppUnitTest.new(test_name,
                                      text_including_current_test[0...current_test_length-1]))
      end
    end
  end

  def find_closing_curly_bracket(text)
    open_curly_brackets=1
    loop do
      if close_bracket_match = @@curly_close_bracket_pattern.match(text)
        unless close_bracket_match.pre_match =~ @@curly_open_bracket_pattern then
          text=close_bracket_match.post_match
          open_curly_brackets -= 1
          return text if open_curly_brackets == 0
          next
        end
      end
      if text =~ @@curly_open_bracket_pattern
          text=$'
          open_curly_brackets += 1
      end
    end
  end

  def identify_statements
    for test in @tests
      statements = Array.new
      sanitized_text = test.body.clone
      while sanitized_text.sub!(@@cpp_loop_construct,'')
        sanitized_text[-find_closing_curly_bracket($').length-1] = " "
      end
      sanitized_text.gsub!(@@cpp_comment_line,'')
      while comma_match = sanitized_text.match(/;/)
        sanitized_text=comma_match.post_match
        type =  case comma_match.pre_match.strip
                  when @@cpp_double_array_definition then :double_array_definition
                  when @@cpp_int_array_definition then :int_array_definition
                  when @@cpp_bool_array_definition then :bool_array_definition
                  when @@cpp_double_definition then :double_definition
                  when @@cpp_int_definition then :int_definition
                  when @@cpp_bool_definition then :bool_definition
                  when @@cpp_delete then :delete
                  when @@gtest_assertion then :gtest_assertion
                  when @@cpp_function_call then :function_call
                  when @@cpp_assignment then :assignment
                  when @@cpp_array_assignment then :array_assignment
                  else :unknown_statement_type
                end
        statements.push(CppStatement.new(type,comma_match.pre_match.strip))
      end
      test.statements = statements
    end
  end
end

class CppToPythonConverter < CppParser

  def initialize(text)
    super(text)
  end

  def convert_cpp_tests_to_python
    python_unit_tests=""
    for test in @tests
      python_unit_tests+=convert_cpp_test_to_python(test)
    end
    return python_unit_tests
  end

  def convert_cpp_test_to_python(test)
    python_unit_test="\n\tdef test" + test.name.sub(/(t|T)est/,"") + "(self):\n\t\t"
    for statement in test.statements
      case statement.type
        when :double_array_definition
          python_unit_test+=convert_array(statement.content,type=:float64)
        when :int_array_definition
          python_unit_test+=convert_array(statement.content,type=:int32)
        when :bool_array_definition
          python_unit_test+=convert_array(statement.content,type=:int32)
        when :double_definition
          python_unit_test+=statement.content.gsub(/double/,"").squeeze(" ").strip
        when :int_definition
          python_unit_test+=statement.content.gsub(/int/,"").squeeze(" ").strip
        when :bool_definition
          python_unit_test+=statement.content.gsub(/bool/,"").gsub(/true/,"True").gsub(/false/,"False").squeeze(" ").strip
        when :delete then next
        when :gtest_assertion
          python_unit_test+="PROCESS BY HAND=>?np.testing.assert_array_equal.?"+statement.content.gsub(/true/,"True").gsub(/false/,"False").strip
        when :function_call
          python_unit_test+="PROCESS BY HAND=>"+statement.content.gsub(/true/,"True").gsub(/false/,"False").strip
        when :assignment
          python_unit_test+=statement.content.gsub(/true/,"True").gsub(/false/,"False").squeeze(" ").strip
        when :array_assignment
          python_unit_test+=statement.content.gsub(/true/,"True").gsub(/false/,"False").squeeze(" ").strip
        when :unknown_statement_type
          python_unit_test+="PROCESS BY HAND=>"+statement.content.strip
        else raise "Incorrectly labelled statement"
      end
      python_unit_test+="\n\t\t"
    end
    return python_unit_test
  end

  def convert_array(array_statement,type)
    array_statement.sub!(/^\s*(int|double|bool)\s*\*\s*(\w+)\s*=/,"\\2 =")
    if array_statement.sub!(/^\s*(\w+)\s*=\s*new\s*(int|double|bool)\[([*0-9a-zA-Z_]+)\]\s*\{\s*(true|false|\d+)\s*\}/,
        "\\1 = np.zeros\(\(\\3\),dtype=np.#{type}\)\n \\1 = \\4")
      array_statement.sub!(/\n\s*\w+\s*=\s*(0|false|0.0)\s*$/,"")
    else
      array_statement.sub!(/\s*new\s*(int|double|bool)\[[*0-9a-zA-Z_]+\]\s*\{\s*(\d|true|false|-)/," np.array\(\[\[\\2")
      if array_statement =~ /\[\[/
        spaces="\t\t"+" "*($`.length+1)
        array_statement.gsub!(/(\d|true|false),\s*\n\s*(\d|-|true|false)/,"\\1\],\n#{spaces}\[\\2")
        array_statement.gsub!(/(\d|true|false)\s*,?\s*\}/,"\\1\]\],\n#{spaces}dtype=np.#{type}\)")
      end
    end
    while array_statement.sub!(/(\s+|\[|,)true(\s+|\]|,)/,"\\1True\\2"); end
    while array_statement.sub!(/(\s+|\[|,)false(\s+|\]|,)/,"\\1False\\2"); end
    return array_statement
  end
end
