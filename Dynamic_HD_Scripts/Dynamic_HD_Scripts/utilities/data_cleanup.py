'''
A module with functions to cleanup of the data directory and prepare
files for deletion by separating them in a specified folder

Created on Jan 11, 2018

@author: thomasriddick
'''

from Dynamic_HD_Scripts.context import workspace_dir
import os
import re

data_dir = "/Users/thomasriddick/Documents/data/HDdata"
generated_file_pattern_202x = re.compile(r'(20(1|2)[0-9]{5}_[0-9]{6})')
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
    exceptions_to_times_to_save = ["20210205_135817","20201123_200519","20221115_190605",
                                   "20221009_114734","20230326_152521","20230218_010605"
                                   "20200726_181304","20221116_113050","20230218_010439",
                                   "20210205_151552","20160714_121938","20171015_031541",
                                   "20170608_140500","20160718_114402","20170419_125745",
                                   "20170420_114224","20170612_202721","20160930_001057",
                                   "20170420_115401","20170612_202559","20170116_123858"]
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

    for dirpath,unused_dirnames,filenames in os.walk(os.path.join(workspace_dir,"Dynamic_HD_Scripts","Dynamic_HD_Scripts")):
        for filename in filenames:
            filepath = os.path.join(dirpath,filename)
            partial_date = None
            if (filename == ".DS_Store" or filename.endswith(".pyc") or
                filename.endswith(".o") or filename.endswith(".so") or
                filename.endswith(".mod") or
                filename == "data_cleanup.py"):
                continue
            with open(filepath,'r') as f:
                for line in f:
                    if partial_date:
                        modified_line = partial_date + line.lstrip(r' "\'')
                        partial_date = None
                    else:
                        modified_line = line
                    while True:
                        match = generated_file_pattern_202x.search(modified_line)
                        if match:
                            creation_times_to_save.append(match.group(0))
                            modified_line = \
                                generated_file_pattern_202x.sub("",modified_line,
                                                                count=1)
                        else:
                            break
                    partial_match = partial_date_pattern.search(modified_line)
                    if partial_match:
                        partial_date = partial_match.group(1)
    for dirpath,unused_dirnames,filenames in os.walk(os.path.join(workspace_dir,"Dynamic_HD_Scripts","tests")):
        for filename in filenames:
            filepath = os.path.join(dirpath,filename)
            partial_date = None
            if (filename == ".DS_Store" or filename.endswith(".pyc") or
                filename.endswith(".o") or filename.endswith(".so") or
                filename.endswith(".mod")):
                continue
            with open(filepath,'r') as f:
                for line in f:
                    if partial_date:
                        modified_line = partial_date + line.lstrip(r' "\'')
                        partial_date = None
                    else:
                        modified_line = line
                    while True:
                        match = generated_file_pattern_202x.search(modified_line)
                        if match:
                            creation_times_to_save.append(match.group(0))
                            modified_line = \
                                generated_file_pattern_202x.sub("",modified_line,
                                                                count=1)
                        else:
                            break
                    partial_match = partial_date_pattern.search(modified_line)
                    if partial_match:
                        partial_date = partial_match.group(1)
    creation_times_to_save = list(set(creation_times_to_save))
    for exception_to_times_to_save in exceptions_to_times_to_save:
        if exception_to_times_to_save in creation_times_to_save:
            creation_times_to_save.remove(exception_to_times_to_save)
    print(creation_times_to_save)
    for directory in directories_to_clean_with_full_path:
        for dirpath,unused_dirnames,filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath,filename)
                match = generated_file_pattern_202x.search(filepath)
                if match:
                    if (not (match.group(0) in creation_times_to_save) or
                        filepath.endswith(".dat")):
                        new_filepath = os.path.join(files_to_delete_directory,
                                                    os.path.relpath(filepath,
                                                                    data_dir))
                        print("Moving: {0}".format(filepath))
                        os.rename(filepath,new_filepath)

if __name__ == '__main__':
    prepare_data_cleanup()
