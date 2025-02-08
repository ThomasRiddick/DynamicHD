import tkinter as tk
from tkinter import ttk

class Counter:

  def __init__(self):
    self.count = -1

  def reset(self):
    self.count = -1

  def __call__(self):
    self.count += 1
    return self.count 

class LakeAnalysisPlotsGui:

  def __init__(self):
    self.corrections_editor_frames = []
    self.show_correction_editor_var = None 

  def add_tab(self,notebook,position,label):
    frame = ttk.Frame(notebook)
    frame.pack()
    notebook.add(frame,text=label)
    return frame

  def add_string_input(self,frame,column,row,label):
    frame_for_element = ttk.Frame(frame)
    frame_for_element.grid(column=column,row=row)
    label = ttk.Label(frame_for_element,text=label)
    entry = ttk.Entry(frame_for_element)
    label.grid(column=0,row=0)
    entry.grid(column=1,row=0)

  def add_filename_input(self,frame,column,row,label):
    self.add_string_input(frame,column,row,label)

  def add_integer_input(self,frame,column,row,label):
    frame_for_element = ttk.Frame(frame)
    frame_for_element.grid(column=column,row=row)
    label = ttk.Label(frame_for_element,text=label)
    entry = ttk.Entry(frame_for_element)
    label.grid(column=0,row=0)
    entry.grid(column=1,row=0)

  def add_boolean_input(self,frame,column,row,label,
                        variable,command=None):
    frame_for_element = ttk.Frame(frame)
    frame_for_element.grid(column=column,row=row)
    box = ttk.Checkbutton(frame_for_element,variable=variable)
    if command is not None:
      box.configure(command=command)
    box.grid(column=0,row=0)
    label = ttk.Label(frame_for_element,text=label)
    label.grid(column=1,row=0)
  
  def add_version_selector(self,frame,column,row,label):
    frame_for_element = ttk.Frame(frame)
    frame_for_element.grid(column=column,row=row)
    label = ttk.Label(frame_for_element,text=label)
    label.grid(column=0,row=0)
    frame_lv = ttk.LabelFrame(frame_for_element)
    frame_lv.grid(column=1,row=0)
    radio_lv = ttk.Radiobutton(frame_lv,text="User latest version")
    radio_lv.grid(column=0,row=0)
    frame_uv = ttk.LabelFrame(frame_for_element)
    frame_uv.grid(column=2,row=0)
    radio_uv = ttk.Radiobutton(frame_uv,text="User Version")
    radio_uv.grid(column=0,row=0)
    self.add_integer_input(frame_uv,column=1,row=0,label=None)
  
  def add_button(self,frame,column,row,label,command):
    button = ttk.Button(frame,text=label,command=command)
    button.grid(column=column,row=row) 
  
  def add_dynamic_label(self,frame,column,row,label):
    frame_for_element = ttk.Frame(frame)
    frame_for_element.grid(column=column,row=row)
    label = ttk.Label(frame_for_element,text=label)
    label.grid(column=0,row=0)
  
  def add_configure_and_save_buttons(self,frame,column,row,
                                     config_command,save_command):
    frame_for_element = ttk.Frame(frame)
    frame_for_element.grid(column=column,row=row) 
    self.add_button(frame_for_element,0,0,label="Configure",command=config_command)
    self.add_button(frame_for_element,1,0,label="Save",command=save_command)
  
  def add_timestep_selector(self,frame,column,row):
    frame_for_element = ttk.Frame(frame)
    frame_for_element.grid(column=column,row=row) 
    counter = Counter()
    self.add_dynamic_label(frame_for_element,counter(),0,label="[MIN TIME]")
    self.add_button(frame_for_element,counter(),0,label="<",command=lambda:0)
    self.add_button(frame_for_element,counter(),0,label="<<",command=lambda:0)
    self.add_button(frame_for_element,counter(),0,label="E<",command=lambda:0)
    self.add_dynamic_label(frame_for_element,counter(),0,label="[INSERT CUR TIME]")
    self.add_button(frame_for_element,counter(),0,label=">",command=lambda:0)
    self.add_button(frame_for_element,counter(),0,label=">>",command=lambda:0)
    self.add_button(frame_for_element,counter(),0,label=">E",command=lambda:0)
    self.add_dynamic_label(frame_for_element,counter(),0,label="[MAX TIME]")
  
  def add_cumulative_flow_slider(self,frame,column,row):
    frame_for_element = ttk.Frame(frame)
    frame_for_element.grid(column=column,row=row) 
    label = ttk.Label(frame_for_element,text="Cumulative Flow")
    label.grid(column=0,row=0)
    slider = ttk.LabeledScale(frame_for_element,from_=0,to=100)
    slider.grid(column=1,row=0)
  
  def add_height_slider(self,frame,column,row,label):
    frame_for_element = ttk.Frame(frame)
    frame_for_element.grid(column=column,row=row) 
    counter = Counter()
    label = ttk.Label(frame_for_element,text=label)
    label.grid(column=counter(),row=0)
    slider = ttk.LabeledScale(frame_for_element,from_=0,to=9000)
    slider.grid(column=counter(),row=0)
    self.add_button(frame_for_element,counter(),0,label="<",command=lambda:0)
    self.add_button(frame_for_element,counter(),0,label="<<",command=lambda:0)
    entry = ttk.Entry(frame_for_element)
    entry.grid(column=counter(),row=0)
    self.add_button(frame_for_element,counter(),0,label=">",command=lambda:0)
    self.add_button(frame_for_element,counter(),0,label=">>",command=lambda:0)

  def add_corrections_editor(self,frame,column,row):
    corrections_editor_frame = ttk.Frame(frame)
    corrections_editor_frame.grid(column=column,row=row)
    label = ttk.Label(corrections_editor_frame,text="Corrections Editor")
    label.grid(column=0,row=0)
    self.add_boolean_input(corrections_editor_frame,1,0,"Select Coordinates",
                           tk.BooleanVar())
    self.add_dynamic_label(corrections_editor_frame,2,0,
                           label="Coords: lat= [COORDS] lon=[COORDS]")
    self.add_dynamic_label(corrections_editor_frame,3,0,"Original Height: [HEIGHT]")
    self.add_integer_input(corrections_editor_frame,4,0,"Adjusted Height")
    self.add_integer_input(corrections_editor_frame,5,0,"Up until date (exclusive):")
    self.add_button(corrections_editor_frame,6,0,label="Write",command=lambda:0)
    self.add_dynamic_label(corrections_editor_frame,7,0,"")
    corrections_editor_frame.grid_remove()
    self.corrections_editor_frames.append(corrections_editor_frame)

  def setup_config_frame(self,config_frame,plots_frame):    
    button = ttk.Button(config_frame,text="Back",
                        command=plots_frame.tkraise)
    button.grid(column=0,row=0)

  def setup_plot_config_selector(self,config_frame,column,row,use_maps_format):
    plot_config_selector_frame = ttk.Frame(config_frame)
    plot_config_selector_frame.grid(column=column,row=row)
    layout_opts = [1,2,4,6] if use_maps_format else [1,2,3,4] 
    drop_down_menus_frames = []
    layout_selector_frame = ttk.Frame(plot_config_selector_frame)
    layout_selector_frame.grid(column=0,row=0)
    num_plots_var = tk.IntVar(value=0)
    for i,opt in enumerate(layout_opts):
      radio = ttk.Radiobutton(layout_selector_frame,text=opt,value=i,
                              variable=num_plots_var,
                              command=
                              lambda:drop_down_menus_frames[\
                                      num_plots_var.get()].tkraise())
      radio.grid(column=i,row=0)
    drop_down_menus_container = ttk.Frame(plot_config_selector_frame)
    drop_down_menus_container.grid(column=0,row=1)
    drop_down_menus_container.columnconfigure(0,weight=1)
    drop_down_menus_container.rowconfigure(0,weight=1)
    for i in layout_opts:
      drop_down_menus_frames.append(
        self.setup_plot_config_drop_down_menus(drop_down_menus_container,
                                               use_maps_format,i))
    drop_down_menus_frames[0].tkraise()

  def setup_plot_config_drop_down_menus(self,drop_down_menus_container,
                                        use_maps_format,num_plots):
    drop_down_menus_frame = ttk.Frame(drop_down_menus_container)
    drop_down_menus_frame.grid(column=0,row=0,sticky="nsew")
    for i in range(num_plots):
        opt_var = tk.StringVar()
        opt_menu = ttk.OptionMenu(drop_down_menus_frame,opt_var)
        if use_maps_format:
          if num_plots > 2:
            j = i*2//num_plots 
            modified_i = i - j*num_plots//2 
          else:
            j = 0
            modified_i = i 
          opt_menu.grid(column=j,row=modified_i)
        else:
          opt_menu.grid(column=0,row=i)
    return drop_down_menus_frame
  
  def add_input_data_panel(self,notebook):
    panel = self.add_tab(notebook,0,"Input Data")
    counter = Counter()
    lake_selector = ttk.Frame(panel)
    lake_selector.grid(column=0,row=counter())
    lakes = ["Lake Agassiz"]
    for i,lake in enumerate(lakes):
      lake_frame = ttk.LabelFrame(lake_selector)
      lake_frame.grid(column=i,row=0)
      radio = ttk.Radiobutton(lake_frame,text=lake)
      radio.grid(column=0,row=0)
    other_lake_frame = ttk.LabelFrame(lake_selector) 
    other_lake_frame.grid(column=len(lakes)+1,row=0)
    radio = ttk.Radiobutton(other_lake_frame,text="Other lake")
    radio.grid(column=0,row=0)
    self.add_integer_input(other_lake_frame,column=1,row=0,label="Min lat")
    self.add_integer_input(other_lake_frame,column=2,row=0,label="Min lon")
    self.add_integer_input(other_lake_frame,column=3,row=0,label="Max lat")
    self.add_integer_input(other_lake_frame,column=4,row=0,label="Max lon")
    self.add_integer_input(panel,0,counter(),"Start date:")
    self.add_integer_input(panel,0,counter(),"End date:")
    self.add_integer_input(panel,0,counter(),"Run Interval:")
    self.add_boolean_input(panel,0,counter(),"Include 0 YBP in list of dates",
                           tk.BooleanVar())
    self.add_filename_input(panel,0,counter(),"Data source 1:")
    self.add_version_selector(panel,0,counter(),"Source 1 version")
    self.add_filename_input(panel,0,counter(),"Data source 2:")
    self.add_version_selector(panel,0,counter(),"Source 2 version")
    self.add_filename_input(panel,0,counter(),"Super fine orography:")
    self.add_string_input( panel,0,counter(),"Glacier mask template")
    button_row = ttk.Frame(panel)
    button_row.grid(column=0,row=counter())
    self.add_button(button_row,0,0,"Update Plots",command=lambda:0)
    self.add_button(button_row,1,0,"Return to current values",command=lambda:0)
    self.add_button(button_row,2,0,"Reset to default values",command=lambda:0)
  
  def add_corrections_editor_panel(self,notebook):
    panel = self.add_tab(notebook,0,"Corrections Editor")
    counter = Counter()
    self.show_correction_editor_var = tk.BooleanVar(value=False)
    self.add_boolean_input(panel,0,counter(),"Show corrections editor",
                           variable=self.show_correction_editor_var,
                           command=self.toggle_corrections_editor)
    self.add_dynamic_label(panel,0,counter(),"Current file: [INSERT FILE]")
    corrs_file_row = ttk.Frame(panel)
    corrs_file_row.grid(column=0,row=counter())
    self.add_filename_input(corrs_file_row,0,0,"Write corrections to file:")
    self.add_button(corrs_file_row,1,0,"Set",command=lambda:0)
    radio_row = ttk.Frame(panel)
    radio_row.grid(column=0,row=counter())
    radio_1 = ttk.Radiobutton(radio_row,text="Corrections for data source 1") 
    radio_1.grid(column=0,row=0)
    radio_2 = ttk.Radiobutton(radio_row,text="Corrections for data source 2") 
    radio_2.grid(column=1,row=0)
  
  def add_time_series_panel(self,notebook):
    panel = self.add_tab(notebook,1,"Time Series")
    panel.columnconfigure(0,weight=1)
    panel.rowconfigure(0,weight=1)
    plots_frame = ttk.Frame(panel) 
    plots_frame.grid(column=0,row=0,sticky="nsew")
    config_frame = ttk.Frame(panel)
    config_frame.grid(column=0,row=0,sticky="nsew")
    plots_frame.tkraise()
    config_row = ttk.Frame(plots_frame)
    config_row.grid(column=0,row=0)
    self.add_configure_and_save_buttons(config_row,0,0,
                                        lambda:config_frame.tkraise(),
                                        lambda:1)
    self.setup_config_frame(config_frame,plots_frame) 
    self.setup_plot_config_selector(config_frame,0,1,use_maps_format=False)
    data_selector_row = ttk.Frame(config_frame)
    data_selector_row.grid(column=0,row=2)
    label = ttk.Label(data_selector_row,text="Agassiz Outlet vs Date")
    label.grid(column=0,row=0)
    data_var = tk.StringVar()
    selector = ttk.OptionMenu(data_selector_row,data_var)
    selector.grid(column=1,row=0)
    
    
  def add_global_maps_panel(self,notebook):
    panel = self.add_tab(notebook,1,"Global Maps")
    panel.columnconfigure(0,weight=1)
    panel.rowconfigure(0,weight=1)
    plots_frame = ttk.Frame(panel) 
    plots_frame.grid(column=0,row=0,sticky="nsew")
    config_frame = ttk.Frame(panel)
    config_frame.grid(column=0,row=0,sticky="nsew")
    plots_frame.tkraise()
    config_row = ttk.Frame(plots_frame)
    config_row.grid(column=0,row=0)
    self.add_configure_and_save_buttons(config_row,0,0,
                                        lambda:config_frame.tkraise(),
                                        lambda:1)
    self.add_timestep_selector(config_row,1,0)
    match_buttons_frame = ttk.Frame(config_row)
    match_buttons_frame.grid(column=2,row=0)
    for i in range(6):
      self.add_button(match_buttons_frame,2+i,0,label=f"Match {i+1}",command=lambda:0)
    self.add_cumulative_flow_slider(config_row,0,1)
    self.add_height_slider(config_row,1,1,"Minimum height")
    self.add_height_slider(config_row,2,1,"Maximum height")
    self.add_corrections_editor(plots_frame,0,1)
    self.setup_config_frame(config_frame,plots_frame) 
    self.setup_plot_config_selector(config_frame,0,1,use_maps_format=True)
  
  def add_lake_maps_panel(self,notebook):
    panel = self.add_tab(notebook,1,"Lake Maps")
    panel.columnconfigure(0,weight=1)
    panel.rowconfigure(0,weight=1)
    plots_frame = ttk.Frame(panel) 
    plots_frame.grid(column=0,row=0,sticky="nsew")
    config_frame = ttk.Frame(panel)
    config_frame.grid(column=0,row=0,sticky="nsew")
    plots_frame.tkraise()
    config_row = ttk.Frame(plots_frame)
    config_row.grid(column=0,row=0)
    self.add_configure_and_save_buttons(config_row,0,0,
                                        lambda:config_frame.tkraise(),
                                        lambda:1)
    self.add_timestep_selector(config_row,1,0)
    self.add_cumulative_flow_slider(config_row,0,1)
    self.add_height_slider(config_row,1,1,"Minimum height")
    self.add_height_slider(config_row,2,1,"Maximum height")
    self.add_corrections_editor(plots_frame,0,1)
    self.setup_config_frame(config_frame,plots_frame) 
    self.setup_plot_config_selector(config_frame,0,1,use_maps_format=True)
  
  def add_cross_sections_panel(self,notebook):
    panel = self.add_tab(notebook,1,"Cross Sections")
    panel.columnconfigure(0,weight=1)
    panel.rowconfigure(0,weight=1)
    plots_frame = ttk.Frame(panel) 
    plots_frame.grid(column=0,row=0,sticky="nsew")
    config_frame = ttk.Frame(panel)
    config_frame.grid(column=0,row=0,sticky="nsew")
    plots_frame.tkraise()
    config_row = ttk.Frame(plots_frame)
    config_row.grid(column=0,row=0)
    self.add_configure_and_save_buttons(config_row,0,0,
                                        lambda:config_frame.tkraise(),
                                        lambda:1)
    self.add_timestep_selector(config_row,1,0)
    self.setup_config_frame(config_frame,plots_frame) 
    data_source_frame = ttk.Frame(config_frame)
    data_source_frame.grid(column=0,row=1)
    radio_1 = ttk.Radiobutton(data_source_frame,text="Show source 1 data") 
    radio_1.grid(column=0,row=0)
    radio_2 = ttk.Radiobutton(data_source_frame,text="Show source 2 data") 
    radio_2.grid(column=1,row=0)
    radio_3 = ttk.Radiobutton(data_source_frame,text="Show data from both sources") 
    radio_3.grid(column=2,row=0)
    spillways_frame = ttk.Frame(config_frame)
    spillways_frame.grid(column=0,row=3)
    radio_4 = ttk.Radiobutton(spillways_frame,text="Show active spillway only") 
    radio_4.grid(column=0,row=0)
    radio_5 = ttk.Radiobutton(spillways_frame,text="Show all potential spillways") 
    radio_5.grid(column=1,row=0)

  def run_gui(self):
    root = tk.Tk()
    outer_frame = ttk.Frame(root)
    outer_frame.grid(column=0,row=0)
    nb = ttk.Notebook(outer_frame)
    nb.grid(column=0,row=0)
    self.add_input_data_panel(nb)
    self.add_time_series_panel(nb)
    self.add_global_maps_panel(nb)
    self.add_lake_maps_panel(nb)
    self.add_cross_sections_panel(nb)
    self.add_corrections_editor_panel(nb)
    root.mainloop()

  def toggle_corrections_editor(self):
    if self.show_correction_editor_var.get():
      for editor_frame in  self.corrections_editor_frames: 
        editor_frame.grid() 
    else:
      for editor_frame in  self.corrections_editor_frames: 
        editor_frame.grid_remove() 

gui = LakeAnalysisPlotsGui()
gui.run_gui()
