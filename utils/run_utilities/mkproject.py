from jinja2 import Environment,FileSystemLoader

rundir = "/Users/thomasriddick/Documents/workspace/Dynamic_HD_Code/run"

def generate_scripts():

    run_script_env = Environment(loader=FileSystemLoader(rundir))
    template  = run_script_env.get_template("prototype")
    print(template.render(input={"slurm_headers":False,
                                 "script_type":"createproject"}))

    print(template.render(input={"slurm_headers":True,
                                 "script_type":"run"}))

    print(template.render(input={"slurm_headers":False,
                             "script_type":"'prerep'"}))

    print(template.render(input={"slurm_headers":False,
                                 "script_type":"'postrep'"}))

generate_scripts()
