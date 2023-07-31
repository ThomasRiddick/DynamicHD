import matplotlib as mpl
import PySimpleGUI as sg
import json
import shutil
import pprint
from Dynamic_HD_Scripts.dynamic_hd_and_dynamic_lake_drivers.dynamic_lake_analysis_driver \
    import Dynamic_Lake_Analysis_Run_Framework

class DynamicLakeAnalysisDriverGUI:

    def __init__(self):
        mpl.use('TkAgg')
        sg.theme("light blue")
        self.layout = self.setup_layout()
        self.config = {}
        self.reopen_load = False

    @staticmethod
    def convert_to_key(name):
        return f'-{name}-'.upper().replace(" ","-").replace("_","-")

    @classmethod
    def get_filepath(cls,name):
       return [sg.Text(name),
               sg.InputText("",key=cls.convert_to_key(name))]

    @classmethod
    def get_string(cls,name):
       return [sg.Text(name),
               sg.InputText("",key=cls.convert_to_key(name))]

    @classmethod
    def get_boolean(cls,name,default):
        return [sg.Checkbox(name,default=default,key=cls.convert_to_key(name))]

    @classmethod
    def get_integer(cls,name,default):
        return [sg.Text(name),
                sg.InputText(str(default),key="-INTEGER-"+cls.convert_to_key(name).lstrip("-"))]

    def setup_layout(self):
        menu_def = [['File',['Exit',]],]
        layout = [[sg.Menu(menu_def)],
                  self.get_filepath("base_directory"),
                  self.get_boolean("setup_directory_structure",False),
                  self.get_filepath("ancillary_data_directory"),
                  self.get_filepath("present_day_base_orography_filepath"),
                  self.get_filepath("base_corrections_filepath"),
                  self.get_filepath("base_true_sinks_filepath"),
                  self.get_string("orography_filepath_template"),
                  self.get_string("landsea_mask_filepath_template"),
                  self.get_string("glacier_mask_filepath_template"),
                  self.get_boolean("generate_lake_orography_corrections",False),
                  self.get_boolean("apply_orography_tweaks",False),
                  self.get_boolean("make_analysis_run",False),
                  self.get_boolean("skip_dynamic_river_production",False),
                  self.get_boolean("skip_dynamic_lake_production",False),
                  self.get_boolean("skip_current_day_time_slice",False),
                  self.get_boolean("run_hd_scripting_default_orography_corrections",False),
                  self.get_integer("start_date",0),
                  self.get_integer("end_date",0),
                  self.get_integer("slice_spacing",10),
                  self.get_boolean("clear_lake_results",False),
                  self.get_boolean("clear_river_results",False),
                  self.get_boolean("clear_river_default_orog_corrs_results",False),
                  self.get_boolean("generate_present_day_rivers_with_original_sink_set",False),
                  self.get_boolean("generate_present_day_rivers_with_true_sinks",False),
                  [sg.Button("Run",key='-RUN-'),sg.Button("Save",key="-SAVE-"),sg.Button("Load",key="-LOAD-")]]
        return layout

    def run_main_event_loop(self):
        self.window = sg.Window("Dynamic Lake Analysis Driver Setup",self.layout,finalize=True,
                                resizable=True)
        while True:
            self.event, self.values = self.window.read()
            if self.event == "-RUN-":
                self.run()
            if self.event == "-SAVE-":
                self.save()
            if self.event == "-LOAD-":
                while True:
                    self.load()
                    if not self.reopen_load:
                        break
            if self.event in (sg.WIN_CLOSED,'Exit'):
                break
        self.window.close()

    def prepare_config(self):
        self.config = {key.strip("-").lower().replace("-","_"):(value if value != "" else None)
                       for key,value in self.values.items() if not isinstance(key,int)}
        keys_of_items_convert = [key for key in self.config.keys() if key.startswith("integer_")]
        for key in keys_of_items_convert:
            #Don't assume python version >= 3.9 (and thus removeprefix)
            self.config[key.replace("integer_","",1)] = int(self.config[key])
            del self.config[key]

    def run(self):
        self.prepare_config()
        driver_object = Dynamic_Lake_Analysis_Run_Framework(**self.config)
        driver_object.run_selected_processes()

    def save(self):
        self.prepare_config()
        with open("/Users/thomasriddick/Documents/data/temp/analysisconfig.json","r") as f:
            configs = json.load(f)
        configs.append(self.config)
        shutil.move("/Users/thomasriddick/Documents/data/temp/analysisconfig.json",
                    "/Users/thomasriddick/Documents/data/temp/analysisconfig.json.bck")
        with open("/Users/thomasriddick/Documents/data/temp/analysisconfig.json","w") as f:
            json.dump(configs,f)

    def load(self):
        self.reopen_load=False
        with open("/Users/thomasriddick/Documents/data/temp/analysisconfig.json","r") as f:
            configs = json.load(f)
        menu_def = [['File',['Exit',]],]
        layout = [[sg.Menu(menu_def)],
                  [sg.Text("Config:"),sg.Spin(list(range(1,1+len(configs))),key='-CONFIG-'),
                   sg.Button("Load"),sg.Button("Delete"),sg.Cancel()]]
        row = []
        for i,config in enumerate(configs):
            row.append(sg.Multiline(str(i+1) + ": " + pprint.pformat(config),
                                    size=(80,25),key='-ML-'+sg.WRITE_ONLY_KEY))
            if (i+1)%2 == 0:
                layout.append(row)
                row = []
        #i.e. if there is a element left over after the loop
        if row:
            layout.append(row)
        loading_window = sg.Window("Load Configuration",layout)
        while True:
            event, values = loading_window.read()
            if event in (sg.WIN_CLOSED,'Exit','Cancel'):
                break
            if event == 'Load':
                self.config = configs[values['-CONFIG-']-1]
                for key,value in self.values.items():
                    if not isinstance(key,int):
                        new_value = self.config[key.strip("-").\
                                                lower().replace("-","_").\
                                                replace("integer_","",1)]
                        if isinstance(new_value,int):
                            new_value = str(new_value)
                        if new_value == "True":
                            new_value = True
                        elif new_value == "False":
                            new_value = False
                        self.window[key].update(new_value)
                break
            if event == 'Delete':
                del_event, _ = sg.Window("Confirm deletion",
                                         [[sg.Text("Proceed with deletion of " + str(values['-CONFIG-']))],
                                          [sg.Button("Confirm"),sg.Cancel()]]).read(close=True)
                if del_event == 'Confirm':
                    with open("/Users/thomasriddick/Documents/data/temp/analysisconfig.json","r") as f:
                        configs = json.load(f)
                    del configs[values['-CONFIG-']-1]
                    shutil.move("/Users/thomasriddick/Documents/data/temp/analysisconfig.json",
                                "/Users/thomasriddick/Documents/data/temp/analysisconfig.json.bck")
                    with open("/Users/thomasriddick/Documents/data/temp/analysisconfig.json","w") as f:
                        json.dump(configs,f)
                self.reopen_load=True
                break
        loading_window.close()

if __name__ == '__main__':
    gui = DynamicLakeAnalysisDriverGUI()
    gui.run_main_event_loop()
