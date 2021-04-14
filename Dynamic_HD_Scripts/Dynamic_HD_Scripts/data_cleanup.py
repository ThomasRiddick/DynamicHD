'''
A module with functions to cleanup of the data directory and prepare
files for deletion by separating them in a specified folder

Created on Jan 11, 2018

@author: thomasriddick
'''

from context import workspace_dir
import os
import re

data_dir = "/Users/thomasriddick/Documents/data/HDdata"
generated_file_pattern_201x = re.compile(r'(201[0-9]{5}_[0-9]{6})')
partial_date_pattern = re.compile(r'([_0-9]+)("|\') *$')
directories_to_clean = ["catchmentmaps","flowmaps","flowparams",
                        "hdfiles/generated","hdrestartfiles/generated",
                        "jsbachrestartfiles/generated","lsmasks/generated",
                        "orogcorrsfields","orographys/generated",
                        "rdirs/generated","rdirs/generated_outflows_marked",
                        "rmouthflow","rmouths","truesinks","updatemasks",
                        "minima","lakeparafiles","basin_catchment_numbers"]
directories_to_clean_with_full_path = [os.path.join(data_dir, directory) for
                                       directory in directories_to_clean]

def prepare_data_cleanup():
    """Prepare to cleanup data by moving all unreference files to a specified folder

    Arguments:None
    Returns: Nothing
    Scan through the dynamic hd code and find all timestamps (including those split over
    two lines) and then moves all files that don't have one of those timestamps to a
    specified folder in order to earmark them for deletion after a final manual check has
    been made that all the files that are still wanted are still in place. Copies the
    internal directory structure of the data directory in the deletion directory so the
    files could be correctly restored to their originally locations if necessary
    """

    creation_times_to_save = []
    files_to_delete_directory = os.path.join(data_dir,"files_to_delete")
    try:
        os.stat(files_to_delete_directory)
    except:
        os.mkdir(files_to_delete_directory)
    for directory in directories_to_clean_with_full_path:
        try:
            os.stat(os.path.join(files_to_delete_directory,os.path.relpath(directory,
                                                                          data_dir)))
        except:
            os.makedirs(os.path.join(files_to_delete_directory,os.path.relpath(directory,
                                                                               data_dir)))
        for unused_dirpath,dirnames,unused_filenames in os.walk(directory):
            for dirname in dirnames:
                try:
                    os.stat(os.path.join(files_to_delete_directory,
                                         os.path.relpath(directory,
                                         data_dir),
                                         dirname))
                except:
                    os.makedirs(os.path.join(files_to_delete_directory,
                                             os.path.relpath(directory,
                                                             data_dir),
                                                             dirname))

    for dirpath,unused_dirnames,filenames in os.walk(workspace_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath,filename)
            partial_date = None
            with open(filepath,'r') as f:
                for line in f:
                    if partial_date:
                        modified_line = partial_date + line.lstrip(r' "\'')
                        partial_date = None
                    else:
                        modified_line = line
                    while True:
                        match = generated_file_pattern_201x.search(modified_line)
                        if match:
                            creation_times_to_save.append(match.group(0))
                            modified_line = \
                                generated_file_pattern_201x.sub("",modified_line,
                                                                count=1)
                        else:
                            break
                    partial_match = partial_date_pattern.search(modified_line)
                    if partial_match:
                        partial_date = partial_match.group(1)

    for directory in directories_to_clean_with_full_path:
        for dirpath,unused_dirnames,filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath,filename)
                match = generated_file_pattern_201x.search(filepath)
                if match:
                    if not (match.group(0) in creation_times_to_save):
                        new_filepath = os.path.join(files_to_delete_directory,
                                                    os.path.relpath(filepath,
                                                                    data_dir))
                        print("Moving: {0}".format(filepath))
                        os.rename(filepath,new_filepath)

if __name__ == '__main__':
    prepare_data_cleanup()
