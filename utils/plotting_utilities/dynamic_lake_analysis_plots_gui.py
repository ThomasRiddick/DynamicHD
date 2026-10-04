import tkinter as tk
from matplotlib.backends.backend_tkagg \
  import FigureCanvasTKAgg, NavigationToolbar2Tk
import copy
import warnings
import os
from tkinter import ttk,filedialog

class Counter:

  def __init__(self):
    self.count = -1

  def reset(self):
    self.count = -1

  def __call__(self):
    self.count += 1
    return self.count 

class MatchZoom:

  def __init__(self,event_handler,index):
    self.event_handler = event_handler
    self.index = index

  def __call__(self):
    self.event_handler.match_zoom(self.index)


class LakeAnalysisPlotsGui:

  def __init__(self,figures,plot_types,timeseries_plot_types,
               initial_configuration,setup_configuration,
               interactive_timeseries_plots,
               interactive_plots,interactive_lake_plots,
               interactive_spillway_plots,
               data_configuration,poll_io_func,
               dbg_plts=None):
    self.figures = figures
    self.plot_types = plot_types
    self.timeseries_plot_types = timeseries_plot_types
    self.initial_configuration = initial_configuration
    self.current_configuration = copy.deepcopy(initial_configuration)
    self.setup_configuration = setup_configuration
    self.dates = []
    self.widget_variables = {}
    self.match_button_frames = {}
    self.timeseries_plots_frames = {}
    self.global_maps_plots_frames = {}
    self.lake_maps_plots_frames = {}
    self.cross_section_plots_frames = {}
    self.corrections_editor_frames = []
    self.interactive_timeseries_plots = interactive_timeseries_plots
    self.interactive_plots = interactive_plots
    self.interactive_lake_plots = interactive_lake_plots
    self.interactive_spillway_plots = interactive_spillway_plots
    self.data_configuration = data_configuration
    self.poll_io_func = poll_io_func
    self.dbg_plts = dbg_plts
    interactive_timeseries_plots.set_coords_and_height_callback_and_tag(
      self.update_height_and_coords_labels,"timeseries")
    interactive_plots.set_coords_and_height_callback_and_tag(
      self.update_height_and_coords_labels,"global_maps")
    interactive_lake_plots.set_coords_and_height_callback_and_tag(
      self.update_height_and_coords_labels,"lake_maps")
    interactive_spillway_plots.set_coords_and_height_callback_and_tag(
      self.update_height_and_coords_labels,"cross_sections")


  def add_tab(self,notebook,position,label):
    frame = ttk.Frame(notebook)
    frame.pack()
    notebook.add(frame,text=label)
    return frame

  def add_string_input(self,name,frame,column,row,label):
    frame_for_element = ttk.Frame(frame)
    frame_for_element.grid(column=column,row=row)
    txt = tk.StringVar()
    label = ttk.Label(frame_for_element,text=label)
    entry = ttk.Entry(frame_for_element,textvariable=txt)
    label.grid(column=0,row=0)
    entry.grid(column=1,row=0)
    self.widget_variables[name] = txt

  def add_filename_input(self,name,frame,column,row,label):
    self.add_string_input(name,frame,column,row,label)

  def add_integer_input(self,name,frame,column,row,label,
                        command=None,command_on_return=None):
    frame_for_element = ttk.Frame(frame)
    frame_for_element.grid(column=column,row=row)
    txt = tk.IntVar(value=0)
    label = ttk.Label(frame_for_element,text=label)
    entry = ttk.Entry(frame_for_element,textvariable=txt)
    if command is not None:
      entry.configure(command=command)
    if command_on_return is not None:
      entry.bind("<Key-Return>",command_on_return)
    label.grid(column=0,row=0)
    entry.grid(column=1,row=0)
    self.widget_variables[name] = txt

  def add_boolean_input(self,name,frame,column,row,label,
                        command=None):
    frame_for_element = ttk.Frame(frame)
    frame_for_element.grid(column=column,row=row)
    flag = tk.BooleanVar(value=False)
    box = ttk.Checkbutton(frame_for_element,variable=flag)
    if command is not None:
      box.configure(command=command)
    box.grid(column=0,row=0)
    label = ttk.Label(frame_for_element,text=label)
    label.grid(column=1,row=0)
    self.widget_variables[name] = flag

  def add_version_selector(self,name,frame,column,row,label):
    frame_for_element = ttk.Frame(frame)
    frame_for_element.grid(column=column,row=row)
    label = ttk.Label(frame_for_element,text=label)
    label.grid(column=0,row=0)
    frame_lv = ttk.LabelFrame(frame_for_element)
    frame_lv.grid(column=1,row=0)
    flag = tk.BooleanVar(default=True)
    radio_lv = ttk.Radiobutton(frame_lv,
                               variable=flag,
                               value=True,
                               text="Use latest version")
    radio_lv.grid(column=0,row=0)
    frame_uv = ttk.LabelFrame(frame_for_element)
    frame_uv.grid(column=2,row=0)
    radio_uv = ttk.Radiobutton(frame_uv,
                               variable=flag,
                               value=False,
                               text="Use version:")
    radio_uv.grid(column=0,row=0)
    self.widget_variables["use_latest_version_for_"+name] = flag
    self.add_integer_input(name+"_fixed_version",
                           frame_uv,column=1,row=0,label=None)

  def add_button(self,frame,column,row,label,command):
    button = ttk.Button(frame,text=label,command=command)
    button.grid(column=column,row=row)

  def add_dynamic_label(self,name,frame,column,row,
                        initial_text):
    frame_for_element = ttk.Frame(frame)
    frame_for_element.grid(column=column,row=row)
    label_var = tk.StringVar(value=initial_text)
    label = ttk.Label(frame_for_element,textvariable=label_var)
    label.grid(column=0,row=0)
    self.widget_variables[name] = label_var

  def add_configure_and_save_buttons(self,frame,column,row,
                                     config_command,save_command):
    frame_for_element = ttk.Frame(frame)
    frame_for_element.grid(column=column,row=row)
    self.add_button(frame_for_element,0,0,
                    label="Configure",command=config_command)
    self.add_button(frame_for_element,1,0,
                    label="Save",command=save_command)

  def add_timestep_selector(self,name,frame,column,row,
                            event_handler):
    frame_for_element = ttk.Frame(frame)
    frame_for_element.grid(column=column,row=row)
    counter = Counter()
    self.add_dynamic_label(name+"_min_date",frame_for_element,
                           counter(),0,"")
    self.add_button(frame_for_element,counter(),0,label="E<",command=lambda:0)
    self.add_button(frame_for_element,counter(),0,label="<<",
                    command=lambda:self.update_timestep(event_handler,-500,
                                                        single_step=False,
                                                        is_relative=True,
                                                        timestep_variable=
                                                        self.widget_variables[name+"_current_date"]))
    self.add_button(frame_for_element,counter(),0,label="<",
                    command=lambda:self.update_timestep(event_handler,-1,
                                                        single_step=True,
                                                        is_relative=True,
                                                        timestep_variable=
                                                        self.widget_variables[name+"_current_date"]))
    self.add_integer_input(name+"_current_date",frame_for_element,counter(),0,"",
                           command_on_return=lambda:self.update_timestep(event_handler,
                                                    self.widget_variables[name+"_current_date"],
                                                    single_step=False,
                                                    is_relative=False,
                                                    timestep_variable=
                                                    self.widget_variables[name+"_current_date"]))
    self.add_button(frame_for_element,counter(),0,label=">",
                    command=lambda:self.update_timestep(event_handler,1,
                                                        single_step=True,
                                                        is_relative=True,
                                                        timestep_variable=
                                                        self.widget_variables[name+"_current_date"]))
    self.add_button(frame_for_element,counter(),0,label=">>",
                    command=lambda:self.update_timestep(event_handler,500,
                                                        single_step=False,
                                                        is_relative=True,
                                                        timestep_variable=
                                                        self.widget_variables[name+"_current_date"]))
    self.add_button(frame_for_element,counter(),0,label=">E",command=lambda:0)
    self.add_dynamic_label(name+"_max_date",frame_for_element,
                           counter(),0,"")

  def add_cumulative_flow_slider(self,name,frame,column,row,event_handler):
    frame_for_element = ttk.Frame(frame)
    frame_for_element.grid(column=column,row=row)
    cumulative_flow_threshold = tk.FloatVar(default=100)
    label = ttk.Label(frame_for_element,variable=cumulative_flow_threshold,
                      text="Cumulative Flow")
    label.grid(column=0,row=0)
    slider = ttk.LabeledScale(frame_for_element,from_=0,to=1000,
                              command=lambda:
                              event_handler.update_minflowcutoff(
                                cumulative_flow_threshold.get()))
    slider.grid(column=1,row=0)
    self.widget_variables[name] = cumulative_flow_threshold

  def add_height_slider(self,tag,name,frame,column,row,label,default,
                        event_handler):
    frame_for_element = ttk.Frame(frame)
    frame_for_element.grid(column=column,row=row)
    height = tk.FloatVar(default=default)
    counter = Counter()
    label = ttk.Label(frame_for_element,text=label)
    label.grid(column=counter(),row=0)
    slider = ttk.LabeledScale(frame_for_element,variable=height,
                              value=default,from_=0.0,to=9000.0,
                              command=lambda:self.update_height(tag,name,
                                                                event_handler))
    slider.grid(column=counter(),row=1)
    self.add_button(frame_for_element,counter(),1,label="<<",
                    command=lambda:self.update_height(tag,name,event_handler,
                                                      to_value=height.get()-100.0))
    self.add_button(frame_for_element,counter(),1,label="<",
                    command=lambda:self.update_height(tag,name,event_handler,
                                                      to_value=height.get()-1.0))
    entry = ttk.Entry(frame_for_element,variable=height,value=default)
    entry.grid(column=counter(),row=1)
    entry.bind("<Key-Return>",lambda event:self.update_height(tag,name,event_handler))
    self.add_button(frame_for_element,counter(),1,label=">",
                    command=lambda:self.update_height(tag,name,event_handler,
                                                      to_value=height.get()+1.0))
    self.add_button(frame_for_element,counter(),1,label=">>",
                    command=lambda:self.update_height(tag,name,event_handler,
                                                      to_value=height.get()+100.0))
    self.widget_variables[name] = height

  def add_corrections_editor(self,name,frame,column,row,event_handler):
    corrections_editor_frame = ttk.Frame(frame)
    corrections_editor_frame.grid(column=column,row=row)
    label = ttk.Label(corrections_editor_frame,text="Correction Editor")
    label.grid(column=0,row=0)
    self.add_boolean_input(name+"_select_coords",corrections_editor_frame,
                           1,0,"Select Coordinates",
                           command=
                           lambda:event_handler.\
                           toggle_select_coords(
                              self.widget_variables[name+"_select_coords"].get()))
    self.add_dynamic_label(name+"_coords",
                           corrections_editor_frame,2,0,
                           "Coords: lat=  lon=")
    self.add_dynamic_label(name+"_original_height",
                           corrections_editor_frame,3,0,
                           "Original Height:")
    self.add_integer_input(name+"_adjust_height",corrections_editor_frame,4,0,
                           "Adjusted Height")
    self.add_integer_input(name+"_up_till_date",
                           corrections_editor_frame,5,0,"Up until date (exclusive):")
    self.add_button(corrections_editor_frame,6,0,label="Write",
                    command=lambda:
                    self.widget_variables[name+"_written"].set(
                    event_handler.\
                    write_correction(new_height=
                                     self.widget_variables[name+"_adjust_height"].get(),
                                     corr_until_date=
                                     self.widget_variables[name+"_up_till_date"].get())))
    self.add_dynamic_label(name+"_written",
                           corrections_editor_frame,7,0,
                           "")
    corrections_editor_frame.grid_remove()
    self.corrections_editor_frames.append(corrections_editor_frame)

  def setup_config_frame(self,config_frame,plots_frame,command=None):
    if command is not None:
      button = ttk.Button(config_frame,text="Back",
                          command=plots_frame.tkraise)
    else:
      def back_callback():
        command()
        plots_frame.tkraise
      button = ttk.Button(config_frame,text="Back",
                          command=back_callback)
    button.grid(column=0,row=0)

  def setup_plot_config_selector(self,name,config_frame,column,row,
                                 use_maps_format,default_selection,
                                 event_handler,plot_types,plots_frames):
    plot_config_selector_frame = ttk.Frame(config_frame)
    plot_config_selector_frame.grid(column=column,row=row)
    layout_opts = [1,2,4,6] if use_maps_format else [1,2,3,4]
    drop_down_menus_frames = []
    layout_selector_frame = ttk.Frame(plot_config_selector_frame)
    layout_selector_frame.grid(column=0,row=0)
    num_plots = tk.IntVar(value=default_selection)
    def change_number_of_plots():
      drop_down_menus_frames[num_plots.get()].tkraise()
      self.select_match_button_set(name)
      self.select_visible_plot(name,plots_frames,use_maps_format)
    for i,opt in enumerate(layout_opts):
      radio = ttk.Radiobutton(layout_selector_frame,text=opt,value=opt,
                              variable=num_plots,
                              command=change_number_of_plots)
      radio.grid(column=i,row=0)
    drop_down_menus_container = ttk.Frame(plot_config_selector_frame)
    drop_down_menus_container.grid(column=0,row=1)
    drop_down_menus_container.columnconfigure(0,weight=1)
    drop_down_menus_container.rowconfigure(0,weight=1)
    for i in layout_opts:
      drop_down_menus_frames.append(
        self.setup_plot_config_drop_down_menus(drop_down_menus_container,
                                               use_maps_format,i,event_handler,
                                               plot_types))
    drop_down_menus_frames[0].tkraise()
    self.widget_variables[name+"_num_plots"] = num_plots

  def setup_plot_config_drop_down_menus(self,drop_down_menus_container,
                                        use_maps_format,num_plots,
                                        event_handler,plot_types):
    drop_down_menus_frame = ttk.Frame(drop_down_menus_container)
    drop_down_menus_frame.grid(column=0,row=0,sticky="nsew")
    #Need new scope so lambda captures correct values
    def setup_plot_config_drop_down_menu(i):
        opt_var = tk.StringVar()
        opt_menu = ttk.Combobox(drop_down_menus_frame,variable=opt_var,
                                values=plot_types)
        opt_menu.bind("<<ComboboxSelected>>",
                      lambda event:event_handler.set_plot_type(f"#{10*num_plots+i}",
                                                               opt_var.get()))
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
    for i in range(num_plots):
      setup_plot_config_drop_down_menu(i)
    return drop_down_menus_frame

  def add_input_data_panel(self,notebook):
    panel = self.add_tab(notebook,0,"Input Data")
    counter = Counter()
    lake_selector = ttk.Frame(panel)
    lake_selector.grid(column=0,row=counter())
    lake_selected = tk.StringVar()
    lakes = ["Lake Agassiz"]
    for i,lake in enumerate(lakes):
      lake_frame = ttk.LabelFrame(lake_selector)
      lake_frame.grid(column=i,row=0)
      radio = ttk.Radiobutton(lake_frame,value=lake,variable=lake_selected,
                              text=lake)
      radio.grid(column=0,row=0)
    other_lake_frame = ttk.LabelFrame(lake_selector)
    other_lake_frame.grid(column=len(lakes)+1,row=0)
    radio = ttk.Radiobutton(other_lake_frame,value="Other lake",variable=lake_selected,
                            text="Other lake")
    radio.grid(column=0,row=0)
    self.widget_variables["lake_selected"] = lake_selected
    self.add_integer_input("min_lat",other_lake_frame,column=1,row=0,label="Min lat")
    self.add_integer_input("min_lon",other_lake_frame,column=2,row=0,label="Min lon")
    self.add_integer_input("max_lat",other_lake_frame,column=3,row=0,label="Max lat")
    self.add_integer_input("max_lon",other_lake_frame,column=4,row=0,label="Max lon")
    self.add_integer_input("start_date",panel,0,counter(),"Start date:",
                           command=self.update_dates)
    self.add_integer_input("end_date",panel,0,counter(),"End date:",
                           command=self.update_dates)
    self.add_integer_input("run_interval",panel,0,counter(),"Run Interval:",
                           command=self.update_dates)
    self.add_boolean_input("include_zero_YBP",panel,0,counter(),
                           "Include 0 YBP in list of dates",
                           command=self.update_dates)
    self.add_filename_input("sequence_one_base_dir",panel,0,
                            counter(),"Data source 1:")
    self.add_version_selector("sequence_one",panel,0,counter(),"Source 1 version")
    self.add_filename_input("sequence_two_base_dir",panel,0,
                            counter(),"Data source 2:")
    self.add_version_selector("sequence_two",panel,0,counter(),"Source 2 version")
    self.add_filename_input("super_fine_orography_filepath",panel,0,
                            counter(),"Super fine orography:")
    self.add_string_input("glacier_mask_file_template",panel,0,
                          counter(),"Glacier mask template")
    button_row = ttk.Frame(panel)
    button_row.grid(column=0,row=counter())
    self.add_button(button_row,0,0,"Update Plots",
                    command=self.update_configuration_and_plots)
    self.add_button(button_row,1,0,"Return to current values",command=
                    lambda:self.set_configuration(self.current_configuration))
    self.add_button(button_row,2,0,"Reset to default values",command=
                    lambda:self.set_configuration(self.initial_configuration))

  def add_corrections_editor_panel(self,notebook):
    panel = self.add_tab(notebook,0,"Correction Editor")
    counter = Counter()
    self.add_boolean_input("corrections_editor_show_editor",
                           panel,0,counter(),"Show corrections editor",
                           command=self.toggle_corrections_editor)
    self.add_dynamic_label("corrections_editor_current_file",
                           panel,0,counter(),"Current file:")
    corrs_file_row = ttk.Frame(panel)
    corrs_file_row.grid(column=0,row=counter())
    self.add_filename_input("corrections_editor_new_file",corrs_file_row,0,0,
                            "Write corrections to file:")
    self.add_button(corrs_file_row,1,0,"Set",command=self.update_corrections_file)
    correct_data_source = tk.IntVar()
    radio_row = ttk.Frame(panel)
    radio_row.grid(column=0,row=counter())
    radio_1 = ttk.Radiobutton(radio_row,value=1,variable=correct_data_source,
                              text="Corrections for data source 1",
                              command=self.update_corrections_data_source)
    radio_1.grid(column=0,row=0)
    radio_2 = ttk.Radiobutton(radio_row,value=2,variable=correct_data_source,
                              text="Corrections for data source 2",
                              command=self.update_corrections_data_source)
    radio_2.grid(column=1,row=0)
    self.widget_variables["corrections_editor_correct_data_source"] = correct_data_source

  def add_timeseries_panel(self,notebook):
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
                                        lambda:self.save_figure("timeseries",
                                          self.figures["timeseries_canvas_"+
                                                       str(self.widget_variables[
                                                       "timeseries_num_plots"].get())]))
    self.setup_config_frame(config_frame,plots_frame)
    self.setup_plot_config_selector("timeseries",
                                    config_frame,0,1,use_maps_format=False,
                                    default_selection=1,
                                    event_handler=
                                    self.interactive_timeseries_plots,
                                    plot_types=self.timeseries_plot_types,
                                    plots_frames=self.timeseries_plots_frames)
    data_selector_row = ttk.Frame(config_frame)
    data_selector_row.grid(column=0,row=2)
    label = ttk.Label(data_selector_row,text="Agassiz Outlet vs Date")
    label.grid(column=0,row=0)
    data_var = tk.StringVar()
    selector = ttk.Combobox(data_selector_row,variable=data_var,
                            values=[None] +
                              self.data_configuration.\
                                get_datasets_by_type("agassizoutlet-time"))
    def set_data_source():
      self.data_configuration.set_configuration("agassizoutlet-time",
                                                data_var.get())
      self.interactive_timeseries_plots.step()
    selector.bind("<<ComboboxSelected>>",
                  lambda event:set_data_source())
    selector.grid(column=1,row=0)
    for figure in self.figures.keys():
      if figure.startswith("timeseries"):
        plots_inner_frame = ttk.Frame(plots_frame)
        self.embed_figure(figure,plots_inner_frame)
        self.timeseries_plots_frames[figure] = plots_inner_frame
    self.timeseries_plots_frames["timeseries_canvas_1"].grid(column=0,row=0)


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
                                        lambda:self.save_figure("global_maps",
                                          self.figures["global_maps_canvas_"+
                                                       str(self.widget_variables[
                                                           "global_maps_num_plots"].get())]))
    self.add_timestep_selector("global_maps",config_row,1,0,
                               self.interactive_plots)
    match_buttons_outer_frame = ttk.Frame(config_row)
    match_buttons_outer_frame.grid(column=2,row=0)
    for i in range(2,8,2):
      match_buttons_frame = ttk.Frame(match_buttons_outer_frame)
      match_buttons_frame.grid(column=0,row=0)
      for j in range(i):
        self.add_button(match_buttons_frame,2+j,0,label=f"Match {j+1}",command=
                        MatchZoom(self.interactive_plots,{2:1,4:3,6:7}[i]+j))
      self.match_button_frames[f"global_{i}"] = match_buttons_frame
    self.add_cumulative_flow_slider("cumulative_flow_global_maps",config_row,0,1,
                                    self.interactive_plots)
    self.add_height_slider("global_maps","min_height_global_maps",
                           config_row,1,1,"Minimum height",0,
                           self.interactive_plots)
    self.add_height_slider("global_maps","max_height_global_maps",
                           config_row,2,1,"Maximum height",5000,
                           self.interactive_plots)
    self.add_corrections_editor("corrections_editor_global_maps",plots_frame,0,1,
                                self.interactive_plots)
    self.setup_config_frame(config_frame,plots_frame)
    self.setup_plot_config_selector("global_maps",
                                    config_frame,0,1,use_maps_format=True,
                                    default_selection=1,
                                    event_handler=self.interactive_plots,
                                    plot_types=self.plot_types,
                                    plots_frames=self.global_maps_plots_frames)
    for figure in self.figures.keys():
      if figure.startswith("global_maps"):
        plots_inner_frame = ttk.Frame(plots_frame)
        self.embed_figure(figure,plots_inner_frame)
        self.global_maps_plots_frames[figure] = plots_inner_frame
    self.global_maps_plots_frames["global_maps_canvas_1"].grid(column=0,row=0)


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
                                        lambda:self.save_figure("lake_maps",
                                               self.figures["lake_maps_canvas_"+
                                                       str(self.widget_variables[
                                                           "lake_maps_num_plots"].get())]))
    self.add_timestep_selector("lake_maps",config_row,1,0,
                               self.interactive_lake_plots)
    match_buttons_outer_frame = ttk.Frame(config_row)
    match_buttons_outer_frame.grid(column=2,row=0)
    for i in range(2,8,2):
      match_buttons_frame = ttk.Frame(match_buttons_outer_frame)
      match_buttons_frame.grid(column=0,row=0)
      for j in range(i):
        self.add_button(match_buttons_frame,2+j,0,label=f"Match {j+1}",command=
                        MatchZoom(self.interactive_lake_plots,{2:1,4:3,6:7}[i]+j))
      self.match_button_frames[f"lake_{i}"] = match_buttons_frame
    self.add_cumulative_flow_slider("cumulative_flow_lake_maps",config_row,0,1,
                                    self.interactive_lake_plots)
    self.add_height_slider("lake_maps","min_height_lake_maps",config_row,1,1,"Minimum height",0,
                           self.interactive_lake_plots)
    self.add_height_slider("lake_maps","max_height_lake_maps",config_row,2,1,"Maximum height",5000,
                           self.interactive_lake_plots)
    self.add_corrections_editor("corrections_editor_lake_maps",plots_frame,0,1,
                                self.interactive_lake_plots)
    self.setup_config_frame(config_frame,plots_frame)
    self.setup_plot_config_selector("lake_maps",
                                    config_frame,0,1,use_maps_format=True,
                                    default_selection=1,event_handler=
                                    self.interactive_lake_plots,
                                    plot_types=self.plot_types,
                                    plots_frames=self.lake_maps_plots_frames)
    for figure in self.figures.keys():
      if figure.startswith("lake_maps"):
        plots_inner_frame = ttk.Frame(plots_frame)
        self.embed_figure(figure,plots_inner_frame)
        self.lake_maps_plots_frames[figure] = plots_inner_frame
    self.lake_maps_plots_frames["lake_maps_canvas_1"].grid(column=0,row=0)

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
                                        lambda:self.save_figure("cross_sections",
                                          self.figures["cross_sections_canvas_"+
                                          "2" if self.widget_variables[
                                          "cross_sections_data_source"].get() == 0
                                          else "1"]))
    self.add_timestep_selector("cross_sections",config_row,1,0,
                               self.interactive_spillway_plots)
    self.setup_config_frame(config_frame,plots_frame,
                            command=self.update_cross_sections_plots())
    source = tk.IntVar()
    active_only = tk.BooleanVar()
    data_source_frame = ttk.Frame(config_frame)
    data_source_frame.grid(column=0,row=1)
    radio_1 = ttk.Radiobutton(data_source_frame,value=1,variable=source,
                              text="Show source 1 data")
    radio_1.grid(column=0,row=0)
    radio_2 = ttk.Radiobutton(data_source_frame,value=2,variable=source,
                              text="Show source 2 data")
    radio_2.grid(column=1,row=0)
    radio_3 = ttk.Radiobutton(data_source_frame,value=0,variable=source,
                              text="Show data from both sources")
    radio_3.grid(column=2,row=0)
    spillways_frame = ttk.Frame(config_frame)
    spillways_frame.grid(column=0,row=3)
    radio_4 = ttk.Radiobutton(spillways_frame,value=True,variable=active_only,
                              text="Show active spillway only")
    radio_4.grid(column=0,row=0)
    radio_5 = ttk.Radiobutton(spillways_frame,value=False,variable=active_only,
                              text="Show all potential spillways")
    radio_5.grid(column=1,row=0)
    self.widget_variables["cross_sections_data_source"] = source
    self.widget_variables["cross_sections_show_active_only"] = active_only
    for figure in self.figures.keys():
      if figure.startswith("cross_sections"):
        plots_inner_frame = ttk.Frame(plots_frame)
        self.embed_figure(figure,plots_inner_frame)
        self.cross_section_plots_frames[figure] = plots_inner_frame
    self.cross_section_plots_frames["cross_sections_canvas_1"].grid(column=0,row=0)

  def add_menu(self,root):
    mb = tk.Menu(root)
    root.config(menu=mb)
    file = tk.Menu(mb,tearoff=False)
    file.add_command(label="Exit",command=
                     lambda:root.destroy())
    mb.add_cascade(label="File",menu=file)
    if self.dbg_plts is not None:
      debug = tk.Menu(mb,tearoff=False)
      debug.add_command(label="Show Plots",command=
                        self.dbg_plts.show_debugging_plots)
      mb.add_cascade(label="Debug",menu=debug)

  def run_gui(self):
    root = tk.Tk()
    self.add_menu(root)
    outer_frame = ttk.Frame(root)
    outer_frame.grid(column=0,row=0)
    nb = ttk.Notebook(outer_frame)
    nb.grid(column=0,row=0)
    self.add_input_data_panel(nb)
    self.add_timeseries_panel(nb)
    self.add_global_maps_panel(nb)
    self.add_lake_maps_panel(nb)
    self.add_cross_sections_panel(nb)
    self.add_corrections_editor_panel(nb)
    def poll_io():
      self.poll_io_func()
      root.after(3000,poll_io)
    root.after(3000,poll_io)
    root.mainloop()

  def toggle_corrections_editor(self):
    if self.widget_variables["corrections_editor_show_editor"].get():
      for editor_frame in  self.corrections_editor_frames:
        editor_frame.grid()
    else:
      for editor_frame in  self.corrections_editor_frames:
        editor_frame.grid_remove()

  def select_match_button_set(self,name):
    number_of_plots = self.widget_variables[name+"_num_plots"].get()
    if number_of_plots == 2:
      self.match_button_frames[name+"_4"].grid_remove()
      self.match_button_frames[name+"_6"].grid_remove()
      self.match_button_frames[name+"_2"].grid()
    elif number_of_plots == 4:
      self.match_button_frames[name+"_2"].grid_remove()
      self.match_button_frames[name+"_6"].grid_remove()
      self.match_button_frames[name+"_4"].grid()
    elif number_of_plots == 6:
      self.match_button_frames[name+"_2"].grid_remove()
      self.match_button_frames[name+"_4"].grid_remove()
      self.match_button_frames[name+"_6"].grid()

  def select_visible_plot(self,name,plots_frames,
                          use_maps_format):
    number_of_plots = self.widget_variables[name+"_num_plots"].get()
    if use_maps_format:
      if number_of_plots == 1:
        plots_frames[name+"_2"].grid_remove()
        plots_frames[name+"_4"].grid_remove()
        plots_frames[name+"_6"].grid_remove()
        plots_frames[name+"_1"].grid()
      elif number_of_plots == 2:
        plots_frames[name+"_1"].grid_remove()
        plots_frames[name+"_4"].grid_remove()
        plots_frames[name+"_6"].grid_remove()
        plots_frames[name+"_2"].grid()
      elif number_of_plots == 4:
        plots_frames[name+"_1"].grid_remove()
        plots_frames[name+"_2"].grid_remove()
        plots_frames[name+"_6"].grid_remove()
        plots_frames[name+"_4"].grid()
      elif number_of_plots == 6:
        plots_frames[name+"_1"].grid_remove()
        plots_frames[name+"_2"].grid_remove()
        plots_frames[name+"_4"].grid_remove()
        plots_frames[name+"_6"].grid()
    else:
      if number_of_plots == 1:
        plots_frames[name+"_2"].grid_remove()
        plots_frames[name+"_3"].grid_remove()
        plots_frames[name+"_4"].grid_remove()
        plots_frames[name+"_1"].grid()
      elif number_of_plots == 2:
        plots_frames[name+"_1"].grid_remove()
        plots_frames[name+"_3"].grid_remove()
        plots_frames[name+"_4"].grid_remove()
        plots_frames[name+"_2"].grid()
      elif number_of_plots == 3:
        plots_frames[name+"_1"].grid_remove()
        plots_frames[name+"_2"].grid_remove()
        plots_frames[name+"_4"].grid_remove()
        plots_frames[name+"_3"].grid()
      elif number_of_plots == 4:
        plots_frames[name+"_1"].grid_remove()
        plots_frames[name+"_2"].grid_remove()
        plots_frames[name+"_3"].grid_remove()
        plots_frames[name+"_4"].grid()

  def update_dates(self):
    start_date = self.widget_variables["start_date"].get()
    end_date = self.widget_variables["end_date"].get()
    interval = self.widget_variables["run_interval"].get()
    include_zero_YBP = self.widget_variables["include_zero_YBP"].get()
    self.dates = list(range(start_date,end_date,-interval))
    if include_zero_YBP:
      self.dates.append(0)

  def update_date_variables(self):
    self.widget_variables["start_date"].set(self.dates[0])
    self.widget_variables["run_interval"].set(self.dates[0] - self.dates[1])
    if self.dates[-1] == 0:
      self.widget_variables["end_date"].set(self.dates[-2])
      self.widget_variables["include_zero_YBP"].set(True)
    else:
      self.widget_variables["end_date"].set(self.dates[-1])
      self.widget_variables["include_zero_YBP"].set(False)

  def update_configuration_and_plots(self):
    #Dates should already be up to date but call update
    #again for robustness
    self.update_dates()
    self.current_configuration["dates"] = self.dates
    self.current_configuration["sequence_one_base_dir"] = \
      self.widget_variables["sequence_one_base_dir"].get()
    self.current_configuration["sequence_two_base_dir"] = \
      self.widget_variables["sequence_two_base_dir"].get()
    self.current_configuration["glacier_mask_file_template"] = \
      self.widget_variables["glacier_mask_file_template"].get()
    self.current_configuration["super_fine_orography_filepath"] = \
      self.widget_variables["super_fine_orography_filepath"].get()
    self.current_configuration["use_latest_version_for_sequence_one"] = \
      self.widget_variables["use_latest_version_for_sequence_one"].get()
    self.current_configuration["sequence_one_fixed_version"] = \
      self.widget_variables["sequence_one_fixed_version"].get()
    self.current_configuration["use_latest_version_for_sequence_two"] = \
      self.widget_variables["use_latest_version_for_sequence_two"].get()
    self.current_configuration["sequence_two_fixed_version"] = \
      self.widget_variables["sequence_two_fixed_version"].get()
    self.setup_configuration(self.current_configuration,reprocessing=True)
    for tag in ["global_maps","lake_maps","cross_sections"]:
      self.widget_variables[tag+"_min_date"].set(self.dates[0])
      self.widget_variables[tag+"_max_date"].set(self.dates[-1])

  def set_configuration(self,configuration):
    self.dates = configuration["dates"]
    self.widget_variables["sequence_one_base_dir"].\
      set(configuration["sequence_one_base_dir"])
    self.widget_variables["sequence_two_base_dir"].\
      set(configuration["sequence_two_base_dir"])
    self.widget_variables["glacier_mask_file_template"].\
      set(configuration["glacier_mask_file_template"])
    self.widget_variables["super_fine_orography_filepath"].\
      set(configuration["super_fine_orography_filepath"])
    self.widget_variables["use_latest_version_for_sequence_one"].\
      set(configuration["use_latest_version_for_sequence_one"])
    self.widget_variables["sequence_one_fixed_version"].\
      set(configuration["sequence_one_fixed_version"])
    self.widget_variables["use_latest_version_for_sequence_two"].\
      set(configuration["use_latest_version_for_sequence_two"])
    self.widget_variables["sequence_two_fixed_version"].\
      set(configuration["sequence_two_fixed_version"])
    self.update_date_variables()

  def embed_figure(self,figure,frame):
    figure_canvas = FigureCanvasTKAgg(figure,frame)
    figure_canvas.draw()
    navigation_toolbar = NavigationToolbar2Tk(figure_canvas)
    navigation_toolbar.update()
    navigation_toolbar.toolitems = [ti for ti in navigation_toolbar.toolitems
                                    if ti[0] != "Save"]
    widget = figure_canvas.get_tk_widget()
    widget.grid(column=0,row=0)

  def update_timestep(self,event_handler,value,single_step,
                      is_relative,timestep_variable):
    if single_step and value < 0:
      event_handler.step_back()
    elif single_step and value > 0:
      event_handler.step_forward()
    else:
      if is_relative:
        new_timestep = event_handler.get_current_date() + value
      else:
        new_timestep = value
      event_handler.step_to_date(new_timestep)
    timestep_variable.set(event_handler.get_current_date())

  def update_height(self,tag,name,event_handler,to_value=None):
    if to_value is not None:
      self.widget_variables[name].set(to_value)
    event_handler.change_height_range(self.widget_variables["min_height_"+tag].get(),
                                      self.widget_variables["max_height_"+tag].get())

  def update_corrections_file(self):
    new_file = self.widget_variables["corrections_editor_new_file"].get()
    self.interactive_plots.set_corrections_file(new_file)
    self.interactive_lake_plots.set_corrections_file(new_file)
    self.widget_variables["corrections_editor_current_file"].set(new_file)

  def update_cross_sections_plots(self):
    self.interactive_spillway_plots.set_plot_active_only(
      self.widget_variables["cross_sections_show_active_only"].get())
    if self.widget_variables["cross_sections_data_source"].get() == 0:
      self.interactive_spillway_plots.step()
      self.cross_section_plots_frames["cross_sections_canvas_2"].tkraise()
    elif self.widget_variables["cross_sections_data_source"].get() == 1:
      self.interactive_spillway_plots.toggle_plot_one()
      self.cross_section_plots_frames["cross_sections_canvas_1"].tkraise()
    elif self.widget_variables["cross_sections_data_source"].get() == 2:
      self.interactive_spillway_plots.toggle_plot_two()
      self.cross_section_plots_frames["cross_sections_canvas_1"].tkraise()

  def update_corrections_data_source(self):
    self.interactive_plots.toggle_use_orog_one_for_original_height(
        self.widget_variables["corrections_editor_correct_data_source"].get() == 1)
    self.interactive_lake_plots.toggle_use_orog_one_for_original_height(
        self.widget_variables["corrections_editor_correct_data_source"].get() == 1)

  def update_height_and_coords_labels(self,lat,lon,original_height,tag):
    self.widget_variables[tag+"_coords"].set(f"Coords: lat={lat}  lon={lon}")
    self.widget_variables[tag+"_original_height"].set(f"Original Height: {original_height}")

  def save_figure(self,name,figure):
    filename = filedialog.asksaveasfilename(title="Save to File",
                                            filetypes=[("PNG","*.png *.PNG")],
                                            default_extension=".png")
    if filename:
        figure.savefig(filename)
