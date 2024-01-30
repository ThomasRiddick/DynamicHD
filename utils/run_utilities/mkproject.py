from jinja2 import Environment,FileSystemLoader
import subprocess
import re

rundir = "/Users/thomasriddick/Documents/workspace/Dynamic_HD_Code/run"

def generate_scripts():

    with open(rundir+"/prototype","r") as f:
        for line in f:
            match = re.match("^\s*method_to_use\s*=\s*(\w*)",line)
            if match:
                method_to_use = match.group(1)
                break
        atmo_grid_res = None
        if method_to_use == "automatic":
            f.seek(0)
            for line in f:
                match = re.match("^\s*icon_atmo_grid_res\s*=\s*(\w*)",line)
                if match:
                    atmo_grid_res = match.group(1).replace("0","").lower()
                    break
        if not atmo_grid_res:
            raise RuntimeError("Template must specify an atmospheric resolution "
                               "for automatic selection between the low resolution or "
                               "the high resolution method")
    if method_to_use == "automatic":
        if atmo_grid_res in ["r2b3","r2b4","r2b5","r2b6"]:
            high_res = False
        else:
            high_res = True
    elif method_to_use in ["high_res","low_res"]:
        high_res = (method_to_use == "high_res")
    else:
        raise RuntimeError("method_to_use value unknown")
    run_script_env = Environment(loader=FileSystemLoader(rundir))
    template  = run_script_env.get_template("prototype")
    create_project = template.render(input={"slurm_headers":False,
                                            "script_type":"createproject"})
    #subprocess.check_output(create_project,shell=True)

    print(template.render(input={"slurm_headers":True,
                                 "script_type":"run"}))

    # print(template.render(input={"slurm_headers":False,
    #                          "script_type":"'prerep'"}))

    # print(template.render(input={"slurm_headers":False,
    #                              "script_type":"'postrep'"}))

generate_scripts()
