from jinja2 import Environment,FileSystemLoader
import configparser
import subprocess
import re
import argparse
import os.path as path
import os
import stat

def convert_to_dict(config_sections):
    config_dict = {}
    for config_section in config_sections:
        for key in config_section:
            config_dict[key] = config_section[key]
    return config_dict

def generate_scripts(input_config_file,
                     store_scripts_in_project,
                     method_to_use=None,
                     generate_config_template_file=None,
                     config_template_settings={}):
    rundir = os.getcwd()
    templates_env = Environment(loader=FileSystemLoader(path.join(rundir,"templates")))
    if generate_config_template_file is not None:
        cfg_template = templates_env.get_template("config.tmpl")
        with open(path.join(rundir,generate_config_template_file),"w") as f:
            f.write(cfg_template.render(input=config_template_settings))
    else:
        project_name = path.basename(input_config_file).removesuffix(".cfg")
        config = configparser.ConfigParser()
        config.read(path.join(rundir,input_config_file))
        config_dict = convert_to_dict([config["Key Settings"],
                                       config["Other Settings"],
                                       config["Additional User Commands"]])
        method_to_use = config["Other Settings"]["method_to_use"]
        if method_to_use == "automatic":
            atmo_grid_res = config["Key Settings"]["icon_atmo_grid_res"].\
                                lower().replace("r0","r").replace("b0","b")
            if atmo_grid_res in ["r2b3","r2b4","r2b5","r2b6"]:
                high_res = False
            else:
                high_res = True
        elif method_to_use in ["high_res","low_res"]:
            high_res = (method_to_use == "high_res")
        else:
            raise RuntimeError("method_to_use value unknown")
        config_dict["high_res"] = high_res
        if high_res:
            config_dict["slurm_headers"] = True
        run_template  = templates_env.get_template("run.tmpl")
        if store_scripts_in_project:
            cp_template  = templates_env.get_template("createproject.tmpl")
            cp_config_dict = config_dict.copy()
            cp_config_dict["script_path"] = rundir
            cp_config_dict["project_name"] = project_name
            create_project = cp_template.render(input=config_dict)
            print(subprocess.check_output(create_project,shell=True))
            run_script_path = ("/".join(rundir.rstrip("/").split("/")[0:-1]) +
                               "/projects/" + project_name +
                               "/scripts/" + project_name+ ".run")
            config_dict["create_project_at_runtime"] = False
        else:
            run_script_path = path.join(rundir,project_name) + ".run"
            config_dict["create_project_at_runtime"] = True
        with open(run_script_path,"w") as f:
            f.write(run_template.render(input=config_dict))
        os.chmod(run_script_path,os.stat(run_script_path).st_mode|stat.S_IXUSR)

class Arguments:
    pass

def parse_arguments():

    args = Arguments()
    parser = argparse.ArgumentParser(prog='ICON HD Parameter File Script Generation Tool',
                                     description='Generate scripts to produce HD parameter '
                                                 '(hdpara*.nc) files for ICON grids',
                                     epilog='')
    parser.add_argument("input_config_file",
                        metavar='Input Config File',
                        type=str,
                        nargs="?",
                        default=None,
                        help="Relative path of the settings file to use to generate"
                             "the scripts from")
    parser.add_argument("-g","--generate-config-template-file",
                        type=str,
                        nargs="?",
                        default=None,
                        help="Create a config template for config file at the "
                             "specified location")
    parser.add_argument("-s","--store-scripts-in-project",
                        action="store_true",
                        help="When true will create a project folder when the "
                             "templte is parsed and store the projects script "
                             "there - otherwise create a script in the run "
                             "directory and only create a project when the "
                             "script is run")

    parser.parse_args(namespace=args)
    return args

if __name__ == '__main__':
    import sys
    #Parse arguments and then run
    args = parse_arguments()
    generate_scripts(**vars(args))
