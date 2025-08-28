class PythonUnitTest
  attr_reader :name, :body, :statements
  attr_writer :name, :body, :statements
  def initialize(name,body)
    @name=name
    @body=body
    @statements = Array.new
  end
  def to_s()
    "#{@name} #{@body} #{@statements}"
  end
end

class PythonStatement
  attr_reader :type,:content
  attr_writer :type,:content
  def initialize(type,content)
    @type = type
    @content = content
  end
end

class PythonParser
  @@continued_line_pattern=/\\/
  @@open_bracket_pattern=/\(/
  @@close_bracket_pattern=/\)/
  @@open_square_bracket_pattern=/\[/
  @@close_square_bracket_pattern=/\]/
  @@test_pattern=/^\s*def test(\w*)\(self\): */
  @@python_comment_line_pattern=/^\s*\#(.*)/
  @@python_array_definition_pattern=/^\s*(\w*)\s*=\s*np\.array\(/
  @@python_array_of_value_definition_pattern=/^\s*(\w*)\s*=\s*np\.full\(\((.*)\),([0-9.]+)(,.*)?\)/
  @@python_list_definition_pattern=/^\s*\w*\s*=\s*\[/
  @@unit_test_assertion=/^\s*self\.assertEqual\(/
  @@numpy_unit_test_assertion=/^\s*np\.testing\.assert_array_equal\(/
  @@block_indent_pattern=/^(\s*).*/
  @@empty_line_pattern=/^(\s*)$/
  @@function_call_pattern = /(.*)=\s*\w+\.\w+\(/
  @@array_assignment_pattern=/^\s*\w+\[\w+,\w+\]\s*=\s*\w+/
  @@dtype_pattern=/,dtype=np\.(.*)/
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
      test_name=$1
      current_test_length = find_block(@text).length
      @tests.push(PythonUnitTest.new(test_name,
                                     @text[0...current_test_length-1]))
    end
  end

  def find_block(text)
    block_text = ""
    block_indent = nil
    text.each_line do |l|
      if l =~ @@empty_line_pattern then
        block_text += l
        next
      end
      l =~ @@block_indent_pattern
      if ! block_indent
        block_indent = $1
        block_text += l
      else
        if $1.length < block_indent.length
          return block_text
        else
          block_text += l
        end
      end
    end
  end

  def count_open_brackets(text_input,initial_count,bracket_type)
    text = text_input.clone
    open_brackets = initial_count
    case bracket_type
      when :round
        open_bracket_pattern = @@open_bracket_pattern
        close_bracket_pattern = @@close_bracket_pattern
      when :square
        open_bracket_pattern = @@open_square_bracket_pattern
        close_bracket_pattern = @@close_square_bracket_pattern
      else raise "Invalid bracket type"
    end
    loop do
      if close_bracket_match = close_bracket_pattern.match(text)
        unless close_bracket_match.pre_match =~ open_bracket_pattern then
          open_brackets -= 1
          text=close_bracket_match.post_match
          return 0 if open_brackets == 0
        end
      else
        return open_brackets
      end
      if text =~ open_bracket_pattern
        text=$'
        open_brackets += 1
      end
    end
  end

  def identify_statements
    for test in @tests
      statements = Array.new
      in_statement = false
      open_brackets = 0
      statement_text = ""
      type = nil
      bracket_type = nil
      continued_line = nil
      test.body.each_line do |l|
        if l =~ @@continued_line_pattern
          if continued_line
            continued_line += $`
          else
            continued_line = $`
          end
          next
        elsif continued_line
          l = continued_line + l
          l.squeeze! " "
          continued_line = nil
        end
        if in_statement
          if open_brackets > 0
            statement_text += l
            open_brackets = count_open_brackets(l,open_brackets,bracket_type)
            if open_brackets == 0
              in_statement = false
            else
              next
            end
          end
        else
          if case l
            when @@python_comment_line_pattern
              type = :comment_line
              statement_text = l
            when @@empty_line_pattern
              type = :empty_line
              statement_text = ""
            when @@unit_test_assertion
              type = :unit_test_assertion
              bracket_type = :round
              open_brackets = count_open_brackets($',1,bracket_type)
              statement_text = l
              if open_brackets > 0
                in_statement = true
                next
              end
            when @@numpy_unit_test_assertion
              type = :unit_test_assertion
              bracket_type = :round
              open_brackets = count_open_brackets($',1,bracket_type)
              statement_text = l.gsub(/np\.testing\.assert_array_equal/,
                                      "self.assertEqual")
              if open_brackets > 0
                in_statement = true
                next
              end
            when @@python_array_of_value_definition_pattern
              type = :array_of_value
              statement_text = l
            when @@python_array_definition_pattern
              type = :two_dimensional_array_definition
              bracket_type = :round
              open_brackets = count_open_brackets($',1,bracket_type)
              statement_text = l
              if open_brackets > 0
                in_statement = true
                next
              end
            when @@array_assignment_pattern
              type = :array_assignment
              statement_text = l
            when @@python_list_definition_pattern
              type = :one_dimensional_array_definition
              bracket_type = :square
              open_brackets = count_open_brackets($',1,bracket_type)
              statement_text = l.gsub(/=\s*\[/,"= np.array([").chomp + ")"
              if open_brackets > 0
                in_statement = true
                next
              end
            when @@function_call_pattern
              #This match is more general than others so needs to come last
              type = :function_call
              bracket_type = :round
              open_brackets = count_open_brackets($',1,bracket_type)
              statement_text = l
              if open_brackets > 0
                in_statement = true
                next
              end
            else
              type = :unknown_statement_type
              statement_text = l
            end
          end
        end
        statements.push(PythonStatement.new(type,statement_text))
      end
      test.statements = statements
    end
  end
end

class PythonToCppConverter < PythonParser

  def initialize(text)
    super(text)
  end

  def convert_python_tests_to_cpp
    cpp_unit_tests=""
    for test in @tests
      cpp_unit_tests+=convert_python_test_to_cpp(test)
    end
    return cpp_unit_tests
  end

  def convert_python_test_to_cpp(test)
    cpp_unit_test="TEST_F(BasinEvaluationTest,#{test.name}) { "
    for statement in test.statements
      case statement.type
      when :comment_line
        cpp_unit_test+=statement.content.gsub(/#/,"//").squeeze(" ").strip.chomp
      when :empty_line
        cpp_unit_test+=""
      when :unit_test_assertion
        cpp_unit_test+= statement.content.gsub(/True/,"true").\
                        gsub(/self\.assertEqual/,"EXPECT_TRUE").\
                        gsub(/,/," == ").\
                        gsub(/False/,"false").squeeze(" ").strip + ";"
      when :array_of_value
        cpp_unit_test+= convert_array_of_value(statement.content)
      when :two_dimensional_array_definition
        cpp_unit_test+=convert_array(statement.content,2)
      when :one_dimensional_array_definition
        cpp_unit_test+=convert_array(statement.content,1)
      when :array_assignment
       cpp_unit_test+=convert_array_assignment(statement.content)
      when :function_call
        cpp_unit_test+=statement.content.strip + ";"
      when :unknown_statement_type
        cpp_unit_test+="PROCESS BY HAND=>"+statement.content.strip
      else raise "Incorrectly labelled statement"
      end
      cpp_unit_test+="\n  "
    end
    return cpp_unit_test.strip + "\n}\n\n"
  end

  def convert_array(array_statement,dims)
    unless @@python_array_definition_pattern =~ array_statement
      raise "Trying to convert invalid statement to array"
    end
    array_name=$1
    array_content="\{#{$'}".chomp + ";\n"
    array_size = nil
    if dims == 2
      array_dim_size=Math.sqrt(array_content.count(",") + 1).round.to_i
      array_size="#{array_dim_size}*#{array_dim_size}"
    else
      array_size=array_content.count(",") + 1
    end
    array_type = nil
    if array_content =~ /True/ || array_content =~ /False/
      array_type = "bool"
    elsif array_content =~ /\./
      array_type = "double"
    else
      array_type = "int"
    end
    array_content.gsub!(/True/,"true")
    array_content.gsub!(/False/,"false")
    array_content.gsub!(/\[/,"")
    array_content.gsub!(/\]/,"")
    array_content.gsub!(/\)/," }")
    new_array_statement =
      "#{array_type}* #{array_name} = new #{array_type}[#{array_size}] #{array_content}"
    return new_array_statement.chomp
  end

  def convert_array_of_value(array_statement)
    unless @@python_array_of_value_definition_pattern =~ array_statement
      raise "Trying to convert invalid statement to array of value"
    end
    array_name=$1
    array_size_raw=$2
    array_value=$3
    array_type_raw=$4
    array_type = nil
    if array_type_raw =~ @@dtype_pattern
      if $1 == "int32"
        array_type = "int"
      elsif $1 == "bool"
        array_type = "bool"
      else
        array_type = "double"
      end
    else
      array_type = "double"
    end
    array_size=array_size_raw.gsub(/,/,"*")
    new_array_statement = "#{array_type}* #{array_name} = new #{array_type}[#{array_size}];\n" +
                          "  std:fill_n(#{array_name},#{array_size},#{array_value});"
    return new_array_statement
  end

  def convert_array_assignment(array_statement)
    new_array_statement =
      array_statement.gsub(/\[/,"[CONVERT ").gsub(/\]/," CONVERT]").strip.chomp + ";"
    return new_array_statement
  end
end
