from jinja2 import Environment,FileSystemLoader

rundir = "/Users/thomasriddick/Documents/workspace/Dynamic_HD_Code/run"

run_script_env = Environment(loader=FileSystemLoader(rundir))

main_template  = run_script_env.get_template("prototype")
print(main_template.render(input={"slurm_headers":True,
                                  "script_type":"run"}))
