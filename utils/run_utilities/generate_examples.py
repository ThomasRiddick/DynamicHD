import sys
import configparser
import argparse
from mkproject import generate_scripts,convert_to_dict

def generate_examples(examples_config_file):
    config = configparser.ConfigParser()
    config.read(examples_config_file)
    for example in config.sections():
        generate_scripts(input_config_file=None,
                         store_scripts_in_project=False,
                         method_to_use=None,
                         generate_config_template_file=f"examples/{example}_example.cfg",
                         config_template_settings=
                         convert_to_dict([config[example]]))

class Arguments:
    pass

def parse_arguments():

    args = Arguments()
    parser = argparse.ArgumentParser(prog='ICON HD Parameter Example Config Generator',
                                     description='Generate example config files'
                                                 'for mkproject',
                                     epilog='')
    parser.add_argument("examples_config_file",
                        metavar='Examples Config File',
                        type=str,
                        help="File containing the settings for the creation of examples files")

    parser.parse_args(namespace=args)
    return args

if __name__ == '__main__':
    import sys
    #Parse arguments and then run
    args = parse_arguments()
    generate_examples(**vars(args))
