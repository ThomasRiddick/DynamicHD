from jinja2 import Environment,FileSystemLoader
import subprocess
import re
import argparse
import os.path as path
import os

def generate_scripts(input_template_filepath,
                     store_scripts_in_project=True,
                     method_to_use=None,
                     atmo_grid_res=None):
    rundir = os.get_cwd()
    input_template_filepath = path.join(rundir,input_template_filepath)
    if atmo_grid_res:
        atmo_grid_res = atmo_grid_res.replace("0","").lower()
    if not method_to_use or method_to_use == "automatic":
        with open(input_template_filepath,"r") as f:
            if not method_to_use:
                for line in f:
                    match = re.match("^\s*method_to_use\s*=\s*(\w*)",line)
                    if match:
                        method_to_use = match.group(1)
                        break
            if method_to_use == "automatic" and not atmo_grid_res:
                f.seek(0)
                for line in f:
                    match = re.match("^\s*icon_atmo_grid_res\s*=\s*(\w*)",line)
                    if match:
                        atmo_grid_res = match.group(1).replace("0","").lower()
                        break
    if method_to_use == "automatic":
        if not atmo_grid_res:
            usr_in = input("Please enter an intended atmospheric grid resolution so "
                           "the correct method to use can be determined; alternative "
                           "enter 'low_res' or 'high_res' to force the use of the "
                           "low or high resolution methods respectively: ")
            if usr_in in ["high_res","low_res"]:
                method_to_use = usr_in
            else:
                atmo_grid_res = usr_in.replace("0","").lower()
        if atmo_grid_res in ["r2b3","r2b4","r2b5","r2b6"]:
            high_res = False
        else:
            high_res = True
    elif method_to_use in ["high_res","low_res"]:
        high_res = (method_to_use == "high_res")
    else:
        raise RuntimeError("method_to_use value unknown")
    run_script_env = Environment(loader=FileSystemLoader(rundir))
    template  = run_script_env.get_template(path.basename(input_template_filepath))
    if store_scripts_in_project:
        create_project = template.render(input={"slurm_headers":False,
                                                "script_type":"createproject",
                                                "script_path":rundir,
                                                "project_name":path.basename(input_template_filepath)})
        print(subprocess.check_output(create_project,shell=True))
        run_script_path = "/".join(rundir.rstrip("/").split("/")[0:-1])+"/projects/"+path.basename(input_template_filepath)+"/scripts/"+path.basename(input_template_filepath)+".run"
    else:
        run_script_path = path.join(rundir,input_template_filepath)+".run"
    with open(run_script_path,"w") as f:
        f.write(template.render(input={"slurm_headers":True,
                                       "script_type":"run"}))

    # print(template.render(input={"slurm_headers":False,
    #                          "script_type":"'prerep'"}))

    # print(template.render(input={"slurm_headers":False,
    #                              "script_type":"'postrep'"}))

class Arguments:
    pass

def parse_arguments():

    args = Arguments()
    parser = argparse.ArgumentParser(prog='ICON HD Parameter File Script Generation Tool',
                                     description='Generate scripts to produce HD parameter '
                                                 '(hdpara*.nc) files for ICON grids',
                                     epilog='')
    parser.add_argument("input_template_filepath",
                        metavar='Input Template Filepath',
                        type=str,
                        help="Relative of the template file to use to generate"
                             "the scripts from")
    parser.add_argument("-s","--store-scripts-in-project",
                        action="store_true",
                        help="When true will create a project folder when the "
                             "templte is parsed and store the projects script "
                             "there - otherwise create a script in the run "
                             "directory and only create a project when the "
                             "script is run")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-m","--method",
                       dest="method_to_use",
                       type=str,
                       choices=["low_res","high_res","automatic"],
                       default=None,
                       help="Force generation of scripts for a particular method")
    group.add_argument("-r","--resolution",
                       dest="atmo_grid_res",
                       type=str,
                       default=None,
                       help="Specify an atmospheric grid resolution explicitly "
                            "for the determination of which method to use. "
                            "It is generally easier to instead set this in the "
                            "template file rather than via this optional argument")

    parser.parse_args(namespace=args)
    return args

def setup_and_run_basic_river_bifurcation_icon_driver(args):
    driver = BifurcateRiversBasicIconDriver(**vars(args))
    driver.run()

if __name__ == '__main__':
    import sys
    sys.argv.append("prototype")
    sys.argv.append("-s")
    #Parse arguments and then run
    args = parse_arguments()
    generate_scripts(**vars(args))
