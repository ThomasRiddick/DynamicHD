from os import path

def check_input_files(files):
    for file in files:
        if not path.isfile(file):
            raise RuntimeError("Input file missing: {}".format(file))

def check_output_files(files):
    for file in files:
        if not path.isdir(path.dirname(path.abspath(file))):
            raise RuntimeError("Output file target location doesn't exist: {}".\
                               format(path.dirname(path.abspath(file))))
        if path.isfile(file):
            raise RuntimeError("Output file already exists: {}".format(file))
