from os import path

def check_input_files(files):
    for file in files:
        if not path.isfile(file):
            raise RuntimeError("Input file missing: {}".format(file))

def check_output_files(files):
    for file in files:
        if not path.isdir(path.dirname(file)):
            raise RuntimeError("Output file target location doesn't exists: {}".format(file))
        if path.isfile(file):
            raise RuntimeError("Output file already exists: {}".format(file))
