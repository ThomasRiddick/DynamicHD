import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import json
import shutil
import pprint
from Dynamic_HD_Scripts.dynamic_hd_and_dynamic_lake_drivers.dynamic_lake_analysis_driver \
    import Dynamic_Lake_Analysis_Run_Framework

class DynamicLakeAnalysisDriverInterfaceGUI:

  def __init__(self,configs_filepath):
    self.config = {}
    self.configs_filepath = configs_filepath

  def reset_row_count(self):
    self.highest_row_number_reached = 0
  
  def get_next_row(self):
    self.highest_row_number_reached += 1
    return self.highest_row_number_reached

  def setup_string_field(self,name):
    row = self.get_next_row()
    frame = ttk.Frame(self.main_panel)
    frame.grid(column=0,row=self.get_next_row(),sticky='ew')
    frame.columnconfigure(1,weight=1)
    txt = tk.StringVar()
    label = ttk.Label(frame,text=f"{name}:")
    label.grid(column=0,row=row)
    entry = ttk.Entry(frame,textvariable=txt)
    entry.grid(column=1,row=row,sticky='ew')
    self.widget_variables[name] = txt

  def setup_filepath_field(self,name):
    self.setup_string_field(name)

  def setup_boolean_field(self,name,default):
    frame = ttk.Frame(self.main_panel)
    frame.grid(column=0,row=self.get_next_row(),sticky='w')
    flag = tk.BooleanVar(value=default)
    box = ttk.Checkbutton(frame,variable=flag)
    box.grid(column=0,row=0,sticky='w')
    label = ttk.Label(frame,text=name)
    label.grid(column=1,row=0)
    self.widget_variables[name] = flag

  def setup_integer_field(self,name,default):
    row = self.get_next_row()
    frame = ttk.Frame(self.main_panel)
    frame.grid(column=0,row=self.get_next_row(),sticky='ew')
    frame.columnconfigure(1,weight=1)
    txt = tk.IntVar(value=default)
    label = ttk.Label(frame,text=f"{name}:")
    label.grid(column=0,row=row)
    entry = ttk.Entry(frame,textvariable=txt)
    entry.grid(column=1,row=row,sticky='ew')
    self.widget_variables[name] = txt

  def setup_label(self,txt):
    label = ttk.Label(self.main_panel,text=txt)
    label.grid(column=0,row=self.get_next_row(),stick="w")

  def setup_gui(self):
    #Setup root and main panel; prep variables
    self.root = tk.Tk()
    self.root.rowconfigure(0,weight=1)
    self.root.columnconfigure(0,weight=1)
    self.root.minsize(400,700)
    self.load_screen_tl = None
    self.widget_variables = {}
    self.root.geometry('800x700+100+100')
    self.root.title("Dynamic Lake Analysis Driver Setup")
    #Required to display correctly on some systems
    self.main_panel = ttk.Frame(self.root)
    self.main_panel.grid(column=0,row=0,sticky="nsew")
    self.main_panel.columnconfigure(0,weight=1)
    #Setup required fields
    self.reset_row_count()
    self.setup_filepath_field("base_directory"),
    self.setup_boolean_field("setup_directory_structure",False),
    self.setup_filepath_field("ancillary_data_directory"),
    self.setup_filepath_field("present_day_base_orography_filepath"),
    self.setup_filepath_field("base_corrections_filepath"),
    self.setup_filepath_field("base_date_based_corrections_filepath"),
    self.setup_filepath_field("base_additional_corrections_filepath"),
    self.setup_filepath_field("base_true_sinks_filepath"),
    self.setup_string_field("orography_filepath_template"),
    self.setup_string_field("landsea_mask_filepath_template"),
    self.setup_string_field("glacier_mask_filepath_template"),
    self.setup_boolean_field("generate_lake_orography_corrections",False),
    self.setup_boolean_field("apply_orography_tweaks",False),
    self.setup_boolean_field("change_date_based_corrections",False),
    self.setup_boolean_field("make_analysis_run",False),
    self.setup_boolean_field("skip_dynamic_river_production",False),
    self.setup_boolean_field("skip_dynamic_lake_production",False),
    self.setup_boolean_field("skip_current_day_time_slice",False),
    self.setup_boolean_field("run_hd_scripting_default_orography_corrections",False),
    self.setup_integer_field("start_date",0),
    self.setup_integer_field("end_date",0),
    self.setup_integer_field("slice_spacing",10),
    self.setup_label("For clear lake and river result -2 is off, -1 is all, 0+ is version num")
    self.setup_integer_field("clear_lake_results",-2),
    self.setup_integer_field("clear_river_results",-2),
    self.setup_boolean_field("clear_river_default_orog_corrs_results",False),
    self.setup_boolean_field("generate_present_day_rivers_with_original_sink_set",False),
    self.setup_boolean_field("generate_present_day_rivers_with_true_sinks",False)
    #Setup button and start
    frame = ttk.Frame(self.main_panel)
    frame.grid(column=0,row=self.get_next_row(),sticky="w")
    button = ttk.Button(frame,text="Run",command=self.run_on_click)
    button.grid(column=0,row=0,sticky="w")
    button = ttk.Button(frame,text="Save",command=self.save_on_click)
    button.grid(column=1,row=0,sticky="w")
    button = ttk.Button(frame,text="Load",command=self.load_on_click)
    button.grid(column=2,row=0,sticky="w")
    button = ttk.Button(frame,text="Print Command",command=self.print_command_on_click)
    button.grid(column=3,row=0,sticky="w")
    button = ttk.Button(frame,text="Clear",command=self.clear_on_click)
    button.grid(column=4,row=0,sticky="w")
    self.read_config()
    self.default_config = self.config.copy()
    self.root.mainloop()

  def setup_load_screen(self):
    self.load_screen_tl = tk.Toplevel(self.root)  
    self.load_screen_tl.bind("<Destroy>",self.restore_main)
    self.load_screen_tl.columnconfigure(0,weight=1)
    #For better ttk styling on some systems
    self.load_screen_panel = ttk.Frame(self.load_screen_tl)
    self.load_screen_panel.grid(column=0,row=0,sticky="nsew") 
    load_screen_top_frame = ttk.Frame(self.load_screen_panel)
    load_screen_top_frame.grid(column=0,row=0)
    self.selected_config = tk.IntVar(value=1)
    selector = ttk.Spinbox(load_screen_top_frame,
         from_=1,to=len(self.loaded_configs),
         increment=1,
         textvariable=self.selected_config)
    selector.grid(column=0,row=0)
    ldbutton = ttk.Button(load_screen_top_frame,text="Load",
        command=self.load_config_on_click)
    ldbutton.grid(column=1,row=0)
    dlbutton = ttk.Button(load_screen_top_frame,text="Delete",
        command=self.delete_config_on_click)
    dlbutton.grid(column=2,row=0)
    clbutton = ttk.Button(load_screen_top_frame,text="Cancel",
              command=self.cancel_on_click)
    clbutton.grid(column=3,row=0)
    self.setup_configs_display()

  def setup_configs_display(self):
    self.configs_display = ttk.Frame(self.load_screen_panel)
    self.configs_display.grid(column=0,row=1,
                              sticky="nsew")
    for i,config in enumerate(self.loaded_configs):
      label_frame = ttk.LabelFrame(self.configs_display)
      label_frame.grid(row=i//3,column=i%3)
      label = ttk.Label(label_frame,text=
                        str(i+1) + ": " +
      pprint.pformat(config)[:800],
      wraplength="16cm")
      label.grid(column=0,row=0)
  
  def run_on_click(self):
    self.root.withdraw()
    self.read_config()
    driver_object = Dynamic_Lake_Analysis_Run_Framework(**self.config)
    driver_object.run_selected_processes()
    self.root.deiconify()

  def save_on_click(self):
    self.read_config()
    with open(self.configs_filepath,"r") as f:
            self.loaded_configs = json.load(f)
    self.loaded_configs.append(self.config)
    shutil.move(self.configs_filepath,
                self.configs_filepath + ".bck")
    with open(self.configs_filepath,"w") as f:
      json.dump(self.loaded_configs,f)

  def load_on_click(self):
    self.root.withdraw()
    self.load_configs()
    self.setup_load_screen()

  def print_command_on_click(self):
    self.read_config()
    positional_argument = ""
    keyword_arguments = ""
    for name,value in self.config.items():
      if value == "":
        continue
      if name == "base_directory":
        positional_argument = " {0}".format(value)
      elif isinstance(value,bool):
        if value:
          keyword_arguments+=" --{0}".format(name.replace("_","-"))
      else:
        keyword_arguments+=" --{0}={1}".format(name.replace("_","-"),
                                              value)   
    printed_command_tl = tk.Toplevel(self.root)  
    printed_command_frame = ttk.Frame(printed_command_tl)
    printed_command_frame.grid(column=0,row=0)
    txt = tk.Text(printed_command_frame)
    txt.insert("1.0",positional_argument + keyword_arguments)
    txt.grid(column=0,row=0,sticky="nsew")
    
  def clear_on_click(self):
    self.config = self.default_config
    self.set_config()

  def restore_main(self,event):
    if event.widget == self.load_screen_tl:
      self.root.deiconify()

  def load_config_on_click(self):
    self.select_config(self.selected_config.get()-1)
    self.set_config() 
    self.load_screen_tl.destroy()

  def delete_config_on_click(self):
    if messagebox.askokcancel(
  title="Proceed?",
        message="Proceed with deletion of " + 
        str(self.selected_config.get()),
  icon='warning'):
      with open(self.configs_filepath,"r") as f:
        self.loaded_configs = json.load(f)
      del self.loaded_configs[self.selected_config.get()-1]
      shutil.move(self.configs_filepath,
                  self.configs_filepath + ".bck")
      with open(self.configs_filepath,"w") as f:
        json.dump(self.loaded_configs,f)
      self.configs_display.grid_forget()
      self.configs_display.destroy()
      self.setup_configs_display()

  def cancel_on_click(self):
    self.load_screen_tl.destroy()

  def read_config(self):
    for name,var in self.widget_variables.items():
      self.config[name] = var.get()

  def set_config(self):
    for name,value in self.config.items():
      if value is not None:
        self.widget_variables[name].set(value)

  def load_configs(self):
    with open(self.configs_filepath,"r") as f:
      self.loaded_configs = json.load(f)    

  def select_config(self,index):
    self.config = self.loaded_configs[index]

configs_filepath = "/Users/thomasriddick/Documents/data/temp/analysisconfig.json"
gui =  DynamicLakeAnalysisDriverInterfaceGUI(configs_filepath)
gui.setup_gui()
