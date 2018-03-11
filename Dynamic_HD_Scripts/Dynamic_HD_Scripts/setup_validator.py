'''
Created on Jan 14, 2018

@author: thomasriddick
'''

import configparser
import sys
import config_layouts
import inspect

class SetupValidator(object):
    """
    Validates ConfigParser configurations according to the templates given in config_layouts
    to ensure they are a valid setup for HD operations. Decide which HD operator is being used
    and checks it has all the required fields; that those fields values conform to the any 
    patterns specified and relationships between fields conform to any patterns specified. 
    Issues warnings if there are any fields in the configuration that are not required or 
    optional fields of the selected HD operator.
    """
    
    max_layout_depth = 1000

    def __init__(self):
        """Class constructor. Create a ConfigParser option to use."""
        self.config = configparser.ConfigParser()
        
    def read_config(self,configuration_filepath):
        """Read a configuration from a given filepath.
        
        Arguments: 
        configuration_filepath: string; the full path to the configuration file to be read
        Returns: 
        nothing
        """

        print "Reading {0}".format(configuration_filepath)
        self.config.read(configuration_filepath)
        
    def process_config(self):
        """Process the configuration currently held by this object.
        
        Arguments: None
        Returns: The name of the 
        Recursively generates new objects using subclasses of GenericConfig according
        to the values given in the config object being held according to the rules 
        in the current config_layout and then replaces the current config_layout with
        the newly created object. Once a terminal node subclass of GenericConfig is
        reached validate the config and use it to select the appropriate driver for
        the HD operator driver member function to run.
        """

        config_layout = config_layouts.GenericConfig()
        for _ in range(0,self.max_layout_depth):
            self.validate_inputs_fields(config_layout.get_required_input_fields(),
                                        config_layout.get_optional_input_fields())
            self.validate_input_field_value_relationships(config_layout.get_field_value_relationships())
            if config_layout.is_terminal_node():
                self.check_for_unused_fields(config_layout.get_required_input_fields(),
                                             config_layout.get_optional_input_fields())
                return config_layout.get_driver_to_use()
            config_layout = getattr(sys.modules[config_layouts.__name__],
                                    config_layout.get_next_layout_name(self.config))
        raise RuntimeError("Fail to process configuration in given configuration file successfully")
            
    def get_config(self):
        """Config getter"""
        return self.config

    def validate_inputs_fields(self,required_input_fields,optional_input_fields):
        for input_field_section,section_input_fields in required_input_fields:
            if not self.config.has_section(input_field_section):
                raise RuntimeError()
            for input_field in section_input_fields:
                if not self.config.has_option(input_field_section,input_field.name):
                    raise RuntimeError()
                for condition in input_field.conditions:
                    if not condition(self.config.get(input_field_section,input_field.name)):
                        raise RuntimeError()
        for input_field_section,section_input_fields in optional_input_fields:
            if self.config.has_section(input_field_section):
                if self.config.has_option(input_field_section,input_field.name):
                    for condition in input_field.conditions:
                        if not condition(self.config.get(input_field_section,input_field.name)):
                            raise RuntimeError()
                    
    def validate_input_field_value_relationships(self,field_value_relationships):
        for relationship in field_value_relationships:
            if not relationship(self.config):
                raise RuntimeError()
            
    def check_for_unused_feilds(self,required_input_fields,optional_input_fields):
        for section in self.config.sections():
            for option in self.config.options(section):
                if (option not in required_input_fields[section] and
                    option not in optional_input_fields[section]):
                    raise UserWarning("Unusued option {0} specified".format(option))
                
    def print_valid_options(self):
        output_str = ''
        for name,config_layout in inspect.getmembers(config_layouts, inspect.isclass):
            if config_layout.is_terminal_node():
                output_str += "Configuration Name: {0} \n".format(name)
                output_str += 'Required\n'
                for key in config_layout.required_input_fields.keys():
                    output_str += "Section: {0}\n".format(key)
                    for values in config_layout.required_input_fields[key]:
                        for value in values:
                            output_str +=  str(value)
                output_str += 'Optional\n'
                for key in config_layout.optional_input_fields.keys():
                    output_str += "Section: {0}\n".format(key)
                    for values in config_layout.optional_input_fields[key]:
                        for value in values:
                            output_str +=  str(value)
        return output_str