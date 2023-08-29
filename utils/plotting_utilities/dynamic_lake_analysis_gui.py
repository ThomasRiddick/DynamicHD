import matplotlib as mpl
import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
import logging
import copy
import re

logging.basicConfig(level=logging.INFO)

class SetCoordsAndHeight:

        def __init__(self,key_prefix,window):
                self.key_prefix = key_prefix
                self.window = window
                self.lat = None
                self.lon = None
                self.date = None
                self.original_height = None

        def get_stored_values(self):
                return self.lat,self.lon,self.date,self.original_height

        def __call__(self,lat,lon,date,original_height):
                self.lat = lat
                self.lon = lon
                self.date = date
                self.original_height = original_height
                for key_number in ["1","2","4","6"]:
                        self.window[f'-{self.key_prefix}CORRLAT{key_number}-'].\
                                update(value=str(lat))
                        self.window[f'-{self.key_prefix}CORRLON{key_number}-'].\
                                update(value=str(lon))
                        self.window[f'-{self.key_prefix}CORRDATE{key_number}-'].\
                                update(value=date)
                        self.window[f'-{self.key_prefix}CORRHEIGHTOUT{key_number}-'].\
                                update(value=str(original_height))


class DynamicLakeAnalysisGUI:

        combo_events = { f'-{key_prefix}PLOTLIST' + event_label + '-':i for
                             i,event_label in enumerate(["11","21","22",
                                                         "41","42","43","44",
                                                         "61","62","63","64",
                                                         "65","66"]) for key_prefix in ['GM','LM']}
        ts_combo_events = { f'-TSPLOTLIST' + event_label + '-':i for
                                i,event_label in enumerate(["11","21","22",
                                                            "31","32","33",
                                                            "41","42","43","44"])}
        default_corrections_file = ("/Users/thomasriddick/Documents/"
                                    "data/temp/erosion_corrections.txt")

        def __init__(self,avail_plots,avail_ts_plots,initial_configuration,
                     dbg_plts=None):
                mpl.use('TkAgg')
                sg.theme("light blue")
                self.avail_plots = avail_plots
                self.avail_ts_plots = avail_ts_plots
                self.visible_column = {"GM":0, "LM":0, "CS":0, "TS":0}
                self.configuration = initial_configuration
                self.initial_configuration = copy.deepcopy(initial_configuration)
                self.dbg_plts = dbg_plts
                self.layout = self.setup_layout()

        @staticmethod
        def config_and_save_button_factory(key_prefix,key_num):
                return sg.Button('Configure',key=f'-{key_prefix}CONFIGURE{key_num}-'),sg.Button('Save')

        @staticmethod
        def sliders_factory(prefix):
                return (sg.Text('Cumulative Flow'),sg.Slider((0,1000),
                                                             orientation='h',
                                                             key=f'-{prefix}ACCSLIDER-',
                                                             default_value=100,
                                                             enable_events=True),
                        sg.Text('Minimum height'),sg.Slider((0,8000),
                                                            orientation='h',
                                                            key=f'-{prefix}ZMINSLIDER-',
                                                            default_value=0,
                                                            enable_events=True),
                        sg.Text('Maximum height'),sg.Slider((0,8000),
                                                            orientation='h',
                                                            key=f'-{prefix}ZMAXSLIDER-',
                                                            default_value=5000,
                                                            enable_events=True),
                        sg.Text("Tip: Click to sides of \nslider to set precise values"))

        def stepping_buttons_factory(self,key_prefix):
                return (sg.Text(self.configuration["dates"][0],key=f'-{key_prefix}STARTDATE-'),
                        sg.Button('E<',key=f"-{key_prefix}EREWIND-",enable_events=True),
                        sg.Button('<<',key=f"-{key_prefix}FREWIND-",enable_events=True),
                        sg.Button('<', key=f"-{key_prefix}REWIND-",enable_events=True),
                        sg.Text(self.configuration["dates"][0],key=f'-{key_prefix}CURRENTDATE-'),
                        sg.Button('>', key=f"-{key_prefix}FORWARD-",enable_events=True),
                        sg.Button('>>',key=f"-{key_prefix}FFORWARD-",enable_events=True),
                        sg.Button('>E',key=f"-{key_prefix}EFORWARD-",enable_events=True),
                        sg.Text(self.configuration["dates"][-1],key=f'-{key_prefix}ENDDATE-'))

        @staticmethod
        def corrections_editor_factory(key_prefix,key_number):
                return [sg.pin(sg.Column([[sg.Text("Correction Editor"),
                                           sg.Checkbox("Select Coordinates",
                                                       key=f'-{key_prefix}SELECTCOORDS{key_number}-',
                                                       enable_events=True),
                                           sg.Text("Date:"),
                                           sg.Text("",
                                                   key=f'-{key_prefix}CORRDATE{key_number}-'),
                                           sg.Text("Coords: "),
                                           sg.Text("lat= "),
                                           sg.Text("",
                                                   key=f'-{key_prefix}CORRLAT{key_number}-'),
                                           sg.Text("lon= "),
                                           sg.Text("",
                                                   key=f'-{key_prefix}CORRLON{key_number}-'),
                                           sg.Text("Original Height:"),
                                           sg.Text("0",
                                                   key=f'-{key_prefix}CORRHEIGHTOUT{key_number}-'),
                                           sg.Text("Adjusted Height:"),
                                           sg.InputText("0",
                                                        key=f'-{key_prefix}CORRHEIGHTIN{key_number}-'),
                                           sg.Button("Write",
                                                     key=f'-{key_prefix}WRITECORR{key_number}-'),
                                           sg.Text("Written: 0 0 0 0",visible=False,
                                                   key=f'-{key_prefix}WRITTENCORR{key_number}-')]],
                                         key=f"-{key_prefix}CORREDIT{key_number}-",
                                         visible=False))]

        def prepare_time_series(self):
                time_series_tab_layout_main_1 = [[*self.config_and_save_button_factory('TS','1')],
                                                 [sg.Canvas(key=f"-TSCANVAS1-",
                                                            size=(1800,800))],
                                                 [sg.Canvas(key=f"-TSNAVCAN1-")]]

                time_series_tab_layout_main_2 =  [[*self.config_and_save_button_factory('TS','2')],
                                                  [sg.Canvas(key=f"-TSCANVAS2-",
                                                             size=(1800,800))],
                                                  [sg.Canvas(key=f"-TSNAVCAN2-")]]

                time_series_tab_layout_main_3 = [[*self.config_and_save_button_factory('TS','3')],
                                                 [sg.Canvas(key=f"-TSCANVAS3-",
                                                            size=(1800,800))],
                                                 [sg.Canvas(key=f"-TSNAVCAN3-")]]

                time_series_tab_layout_main_4 = [[*self.config_and_save_button_factory('TS','4')],
                                                 [sg.Canvas(key=f"-TSCANVAS4-",
                                                            size=(1800,800))],
                                                 [sg.Canvas(key=f"-TSNAVCAN4-")]]

                time_series_subpanel_layout_configure_1 = [[sg.Combo(self.avail_ts_plots,
                                                                     key=f'-TSPLOTLIST11-',enable_events=True)]]
                time_series_subpanel_layout_configure_2 = [[sg.Combo(self.avail_ts_plots,
                                                                     key=f'-TSPLOTLIST21-',enable_events=True)],
                                                           [sg.Combo(self.avail_ts_plots,
                                                                     key=f'-TSPLOTLIST22-',enable_events=True)]]
                time_series_subpanel_layout_configure_3 = [[sg.Combo(self.avail_ts_plots,
                                                            key=f'-TSPLOTLIST31-',enable_events=True)],
                                                           [sg.Combo(self.avail_ts_plots,
                                                            key=f'-TSPLOTLIST32-',enable_events=True)],
                                                           [sg.Combo(self.avail_ts_plots,
                                                            key=f'-TSPLOTLIST33-',enable_events=True)]]
                time_series_subpanel_layout_configure_4 = [[sg.Combo(self.avail_ts_plots,
                                                            key=f'-TSPLOTLIST41-',enable_events=True)],
                                                           [sg.Combo(self.avail_ts_plots,
                                                            key=f'-TSPLOTLIST42-',enable_events=True)],
                                                           [sg.Combo(self.avail_ts_plots,
                                                            key=f'-TSPLOTLIST43-',enable_events=True)],
                                                           [sg.Combo(self.avail_ts_plots,
                                                            key=f'-TSPLOTLIST44-',enable_events=True)]]

                time_series_tab_layout_configure = [[sg.Button('Back',key=f'-TSBACK-')],
                                                    [sg.Radio('1', f'TSPLOTNUMRADIO',key=f'-TSPLOTNUMRADIO1-',
                                                              default=True,enable_events=True),
                                                     sg.Radio('2', f'TSPLOTNUMRADIO',key=f'-TSPLOTNUMRADIO2-',
                                                              enable_events=True),
                                                     sg.Radio('3', f'TSPLOTNUMRADIO',key=f'-TSPLOTNUMRADIO3-',
                                                              enable_events=True),
                                                     sg.Radio('4', f'TSPLOTNUMRADIO',key=f'-TSPLOTNUMRADIO4-',
                                                              enable_events=True)],
                                                    [sg.Column(time_series_subpanel_layout_configure_1,
                                                               key=f'-TSLC1-',visible=True),
                                                     sg.Column(time_series_subpanel_layout_configure_2,
                                                               key=f'-TSLC2-',visible=False),
                                                     sg.Column(time_series_subpanel_layout_configure_3,
                                                               key=f'-TSLC3-',visible=False),
                                                     sg.Column(time_series_subpanel_layout_configure_4,
                                                               key=f'-TSLC4-',visible=False)]]
                self.visible_column['TS'] = 1

                time_series_tab_layout_wrapper = [[sg.Column(time_series_tab_layout_main_1,
                                                             key=f'-TSLM1-',visible=True),
                                                   sg.Column(time_series_tab_layout_main_2,
                                                             key=f'-TSLM2-',visible=False),
                                                   sg.Column(time_series_tab_layout_main_3,
                                                             key=f'-TSLM3-',visible=False),
                                                   sg.Column(time_series_tab_layout_main_4,
                                                             key=f'-TSLM4-',visible=False),
                                                   sg.Column(time_series_tab_layout_configure,
                                                             key=f'-TSLC-',visible=False)]]
                return time_series_tab_layout_wrapper

        def prepare_maps(self,key_prefix):
                maps_tab_layout_main_1 = [[*self.config_and_save_button_factory(key_prefix,'1'),
                                           *self.stepping_buttons_factory(f"{key_prefix}1"),
                                           *self.sliders_factory(f"{key_prefix}1")],
                                          self.corrections_editor_factory(key_prefix,'1'),
                                          [sg.Canvas(key=f"-{key_prefix}CANVAS1-",
                                                     size=(1800,800))],
                                          [sg.Canvas(key=f"-{key_prefix}NAVCAN1-")]]

                maps_tab_layout_main_2 =  [[*self.config_and_save_button_factory(key_prefix,'2'),
                                           *self.stepping_buttons_factory(f"{key_prefix}2"),
                                           *self.sliders_factory(f"{key_prefix}2")],
                                           self.corrections_editor_factory(key_prefix,'2'),
                                           [sg.Canvas(key=f"-{key_prefix}CANVAS2-",
                                                      size=(1800,800))],
                                           [sg.Canvas(key=f"-{key_prefix}NAVCAN2-")]]

                maps_tab_layout_main_4 = [[*self.config_and_save_button_factory(key_prefix,'4'),
                                           *self.stepping_buttons_factory(f"{key_prefix}4"),
                                           *self.sliders_factory(f"{key_prefix}4")],
                                          self.corrections_editor_factory(key_prefix,'4'),
                                          [sg.Canvas(key=f"-{key_prefix}CANVAS4-",
                                                     size=(1800,800))],
                                          [sg.Canvas(key=f"-{key_prefix}NAVCAN4-")]]

                maps_tab_layout_main_6 = [[*self.config_and_save_button_factory(key_prefix,'6'),
                                           *self.stepping_buttons_factory(f"{key_prefix}6"),
                                           *self.sliders_factory(f"{key_prefix}6")],
                                          self.corrections_editor_factory(key_prefix,'6'),
                                          [sg.Canvas(key=f"-{key_prefix}CANVAS6-",
                                                     size=(1800,800))],
                                          [sg.Canvas(key=f"-{key_prefix}NAVCAN6-")]]

                maps_subpanel_layout_configure_1 = [[sg.Combo(self.avail_plots,
                                                     key=f'-{key_prefix}PLOTLIST11-',enable_events=True)]]
                maps_subpanel_layout_configure_2 = [[sg.Combo(self.avail_plots,
                                                     key=f'-{key_prefix}PLOTLIST21-',enable_events=True)],
                                                    [sg.Combo(self.avail_plots,
                                                     key=f'-{key_prefix}PLOTLIST22-',enable_events=True)]]
                maps_subpanel_layout_configure_4 = [[sg.Combo(self.avail_plots,
                                                     key=f'-{key_prefix}PLOTLIST41-',enable_events=True),
                                                     sg.Combo(self.avail_plots,
                                                     key=f'-{key_prefix}PLOTLIST42-',enable_events=True)],
                                                    [sg.Combo(self.avail_plots,
                                                     key=f'-{key_prefix}PLOTLIST43-',enable_events=True),
                                                     sg.Combo(self.avail_plots,
                                                     key=f'-{key_prefix}PLOTLIST44-',enable_events=True)]]
                maps_subpanel_layout_configure_6 = [[sg.Combo(self.avail_plots,
                                                     key=f'-{key_prefix}PLOTLIST61-',enable_events=True),
                                                     sg.Combo(self.avail_plots,
                                                     key=f'-{key_prefix}PLOTLIST62-',enable_events=True)],
                                                    [sg.Combo(self.avail_plots,
                                                     key=f'-{key_prefix}PLOTLIST63-',enable_events=True),
                                                     sg.Combo(self.avail_plots,
                                                     key=f'-{key_prefix}PLOTLIST64-',enable_events=True)],
                                                    [sg.Combo(self.avail_plots,
                                                     key=f'-{key_prefix}PLOTLIST65-',enable_events=True),
                                                     sg.Combo(self.avail_plots,
                                                     key=f'-{key_prefix}PLOTLIST66-',enable_events=True)]]

                maps_tab_layout_configure = [[sg.Button('Back',key=f'-{key_prefix}BACK-')],
                                             [sg.Radio('1', f'{key_prefix}PLOTNUMRADIO',key=f'-{key_prefix}PLOTNUMRADIO1-',default=True,enable_events=True),
                                              sg.Radio('2', f'{key_prefix}PLOTNUMRADIO',key=f'-{key_prefix}PLOTNUMRADIO2-',enable_events=True),
                                              sg.Radio('4', f'{key_prefix}PLOTNUMRADIO',key=f'-{key_prefix}PLOTNUMRADIO4-',enable_events=True),
                                              sg.Radio('6', f'{key_prefix}PLOTNUMRADIO',key=f'-{key_prefix}PLOTNUMRADIO6-',enable_events=True)],
                                             [sg.Column(maps_subpanel_layout_configure_1,key=f'-{key_prefix}LC1-',visible=True),
                                              sg.Column(maps_subpanel_layout_configure_2,key=f'-{key_prefix}LC2-',visible=False),
                                              sg.Column(maps_subpanel_layout_configure_4,key=f'-{key_prefix}LC4-',visible=False),
                                              sg.Column(maps_subpanel_layout_configure_6,key=f'-{key_prefix}LC6-',visible=False)]]
                self.visible_column[key_prefix] = 1

                maps_tab_layout_wrapper = [[sg.Column(maps_tab_layout_main_1,key=f'-{key_prefix}LM1-',visible=True),
                                            sg.Column(maps_tab_layout_main_2,key=f'-{key_prefix}LM2-',visible=False),
                                            sg.Column(maps_tab_layout_main_4,key=f'-{key_prefix}LM4-',visible=False),
                                            sg.Column(maps_tab_layout_main_6,key=f'-{key_prefix}LM6-',visible=False),
                                            sg.Column(maps_tab_layout_configure,key=f'-{key_prefix}LC-',visible=False)]]
                return maps_tab_layout_wrapper

        def prepare_cross_sections(self):
                cross_sections_tab_layout_configure = [[sg.Button('Back',key='-CSBACK-')],
                                                       [sg.Radio('Show source 1 data','CSRADIO',default=True,
                                                                 key="-CSPLOTOPTRADIO1-"),
                                                        sg.Radio('Show source 2 data','CSRADIO',
                                                                 key="-CSPLOTOPTRADIO2-"),
                                                        sg.Radio('Show data from both sources','CSRADIO',
                                                                 key="-CSPLOTOPTRADIOBOTH-")]]

                cross_sections_tab_layout_main_1 = [[sg.Button('Configure',key='-CSCONFIGURE1-'),
                                                     sg.Button('Save'),*self.stepping_buttons_factory(f"CS1")],
                                                    [sg.Canvas(key=f"-CSCANVAS1-",
                                                                        size=(1800,800))]]
                cross_sections_tab_layout_main_2 = [[sg.Button('Configure',key='-CSCONFIGURE2-'),
                                                     sg.Button('Save'),*self.stepping_buttons_factory(f"CS2")],
                                                    [sg.Canvas(key=f"-CSCANVAS2-",
                                                               size=(1800,800))]]
                self.visible_column["CS"] = 1

                cross_sections_tab_layout_wrapper = [[sg.Column(cross_sections_tab_layout_main_1,key='-CSLM1-'),
                                                      sg.Column(cross_sections_tab_layout_main_2,key='-CSLM2-',visible=False),
                                                      sg.Column(cross_sections_tab_layout_configure,key='-CSLC-',visible=False)]]
                return cross_sections_tab_layout_wrapper

        def change_visible_column(self,key_base,visible_key_num,numeric_labels=[1,2,4,6]):
                for label in numeric_labels:
                        self.window[f'-{key_base}{label}-'].update(visible=(visible_key_num == label))

        def change_correction_editor_visibility(self):
                for prefix in ["GM","LM"]:
                        for label in [1,2,4,6]:
                                self.window[f'-{prefix}CORREDIT{label}-'].update(visible=
                                                                        self.values["-SHOWCORREDITOR-"])

        def process_config_main_switches_for_maps(self,key_prefix):
                if self.event.startswith("-GM") or self.event.startswith("-LM"):
                        if self.event == f"-{key_prefix}BACK-":
                                if self.values[f'-{key_prefix}PLOTNUMRADIO1-']:
                                        self.change_visible_column(f'{key_prefix}LM',1)
                                        self.visible_column[key_prefix] = 1
                                elif self.values[f'-{key_prefix}PLOTNUMRADIO2-']:
                                        self.change_visible_column(f'{key_prefix}LM',2)
                                        self.visible_column[key_prefix] = 2
                                elif self.values[f'-{key_prefix}PLOTNUMRADIO4-']:
                                        self.change_visible_column(f'{key_prefix}LM',4)
                                        self.visible_column[key_prefix] = 4
                                elif self.values[f'-{key_prefix}PLOTNUMRADIO6-']:
                                        self.change_visible_column(f'{key_prefix}LM',6)
                                        self.visible_column[key_prefix] = 6
                                self.window[f'-{key_prefix}LC-'].update(visible=False)
                        if (self.event == f'-{key_prefix}CONFIGURE1-' or self.event == f'-{key_prefix}CONFIGURE2-' or
                            self.event == f'-{key_prefix}CONFIGURE4-' or self.event == f'-{key_prefix}CONFIGURE6-'):
                                self.change_visible_column(f'{key_prefix}LM',0)
                                self.window[f'-{key_prefix}LC-'].update(visible=True)
                                self.visible_column[key_prefix] = 0
                        if self.event == f'-{key_prefix}PLOTNUMRADIO1-':
                                self.change_visible_column(f'{key_prefix}LC',1)
                        if self.event == f'-{key_prefix}PLOTNUMRADIO2-':
                                self.change_visible_column(f'{key_prefix}LC',2)
                        if self.event == f'-{key_prefix}PLOTNUMRADIO4-':
                                self.change_visible_column(f'{key_prefix}LC',4)
                        if self.event == f'-{key_prefix}PLOTNUMRADIO6-':
                                self.change_visible_column(f'{key_prefix}LC',6)

        def process_config_main_switches_for_time_series(self):
                if self.event.startswith("-TS"):
                        if self.event == f"-TSBACK-":
                                if self.values[f'-TSPLOTNUMRADIO1-']:
                                        self.change_visible_column(f'TSLM',1,[1,2,3,4])
                                        self.visible_column['TS'] = 1
                                elif self.values[f'-TSPLOTNUMRADIO2-']:
                                        self.change_visible_column(f'TSLM',2,[1,2,3,4])
                                        self.visible_column['TS'] = 2
                                elif self.values[f'-TSPLOTNUMRADIO3-']:
                                        self.change_visible_column(f'TSLM',3,[1,2,3,4])
                                        self.visible_column['TS'] = 3
                                elif self.values[f'-TSPLOTNUMRADIO4-']:
                                        self.change_visible_column(f'TSLM',4,[1,2,3,4])
                                        self.visible_column['TS'] = 4
                                self.window[f'-TSLC-'].update(visible=False)
                        if (self.event == f'-TSCONFIGURE1-' or self.event == f'-TSCONFIGURE2-' or
                            self.event == f'-TSCONFIGURE3-' or self.event == f'-TSCONFIGURE4-'):
                                self.change_visible_column(f'TSLM',0,[1,2,3,4])
                                self.window[f'-TSLC-'].update(visible=True)
                                self.visible_column['TS'] = 0
                        if self.event == f'-TSPLOTNUMRADIO1-':
                                self.change_visible_column(f'TSLC',1,[1,2,3,4])
                        if self.event == f'-TSPLOTNUMRADIO2-':
                                self.change_visible_column(f'TSLC',2,[1,2,3,4])
                        if self.event == f'-TSPLOTNUMRADIO3-':
                                self.change_visible_column(f'TSLC',3,[1,2,3,4])
                        if self.event == f'-TSPLOTNUMRADIO4-':
                                self.change_visible_column(f'TSLC',4,[1,2,3,4])

        def process_config_main_switches_for_cross_sections(self,event_handler):
                if self.event == "-CSBACK-":
                        if self.values["-CSPLOTOPTRADIO1-"]:
                                self.change_visible_column("CSLM",1,[1,2])
                                self.visible_column["CS"] = 1
                                event_handler.toggle_plot_one()
                        if self.values["-CSPLOTOPTRADIO2-"]:
                                self.change_visible_column("CSLM",1,[1,2])
                                self.visible_column["CS"] = 1
                                event_handler.toggle_plot_two()
                        if self.values["-CSPLOTOPTRADIOBOTH-"]:
                                self.change_visible_column("CSLM",2,[1,2])
                                self.visible_column["CS"] = 2
                                event_handler.replot()
                        self.window['-CSLC-'].update(visible=False)
                if self.event == "-CSCONFIGURE1-" or self.event == "-CSCONFIGURE2-":
                        self.window['-CSLM1-'].update(visible=False)
                        self.window['-CSLM2-'].update(visible=False)
                        self.window['-CSLC-'].update(visible=True)
                        self.visible_column["CS"] = 0

        def check_for_set_combo_events(self,prefix,event_handler):
                if self.event in self.combo_events.keys():
                        if self.event.startswith(f'-{prefix}'):
                                event_handler.set_plot_type(self.combo_events[self.event],
                                                            self.values[self.event])
                if self.event in self.ts_combo_events.keys():
                        if self.event.startswith(f'-{prefix}'):
                                event_handler.set_plot_type(self.ts_combo_events[self.event],
                                                            self.values[self.event])


        def check_for_slider_events(self,prefix,event_handler):
                if self.event.endswith('SLIDER-') and self.event.startswith(f'-{prefix}'):
                        if self.event in [f'-{prefix}{i}ACCSLIDER-' for i in ["1","2","4","6"]]:
                                event_handler.update_minflowcutoff(self.values[self.event])
                        for i in ["1","2","4","6"]:
                                if self.event in [f'-{prefix}{i}ZMINSLIDER-',
                                                  f'-{prefix}{i}ZMAXSLIDER-']:
                                        event_handler.change_height_range(
                                                self.values[f'-{prefix}{i}ZMINSLIDER-'],
                                                self.values[f'-{prefix}{i}ZMAXSLIDER-'])

        def check_for_stepping_events(self,key_prefix,event_handler):
                forward_stepping_events = [f'-{key_prefix}'+ event_label + 'FORWARD' + '-' for
                                           event_label in ["1","2","4","6"]]
                backward_stepping_events = [f'-{key_prefix}'+ event_label + 'REWIND' + '-' for
                                           event_label in ["1","2","4","6"]]
                if self.event in forward_stepping_events:
                        event_handler.step_forward()
                        self.update_date(key_prefix,event_handler)
                if self.event in backward_stepping_events:
                        event_handler.step_back()
                        self.update_date(key_prefix,event_handler)
                fast_forward_stepping_events = [f'-{key_prefix}'+ event_label + 'FFORWARD' + '-' for
                                                event_label in ["1","2","4","6"]]
                fast_backward_stepping_events = [f'-{key_prefix}'+ event_label + 'FREWIND' + '-' for
                                                 event_label in ["1","2","4","6"]]
                if (self.event in fast_forward_stepping_events or
                    self.event in fast_backward_stepping_events):
                        date = (event_handler.get_current_date() +
                                (-500 if (self.event in fast_forward_stepping_events) else 500))
                        event_handler.step_to_date(date)
                        self.update_date(key_prefix,event_handler)

        def check_for_select_coords_events(self,key_prefix,event_handler):
                labels = [f'-{key_prefix}SELECTCOORDS{event_label}-' for
                                           event_label in ["1","2","4","6"]]
                if self.event in labels:
                        event_handler.toggle_select_coords(self.values[self.event])

        def check_for_write_correction_events(self,key_prefix,event_handler):
                labels = [f'-{key_prefix}WRITECORR{event_label}-' for
                                           event_label in ["1","2","4","6"]]
                if self.event in labels:
                        key_number = re.match(f"-{key_prefix}WRITECORR(\d)-",self.event).group(1)
                        event_handler.write_correction(new_height=
                                                       self.values[f'-{key_prefix}CORRHEIGHTIN{key_number}-'])

        def check_for_correction_source_events(self,event_handlers):
                if self.event == "-CORRSOURCERADIO1-" or self.event == "-CORRSOURCERADIO2-":
                        for event_handler in event_handlers:
                                event_handler.\
                                toggle_use_orog_one_for_original_height(self.event ==
                                                                        "-CORRSOURCERADIO1-")

        def check_for_include_threshold_in_corrected_slices_events(self,event_handlers):
                if self.event == "-INCLUDETHRESINCORRS-":
                        for event_handler in event_handlers:
                                event_handler.\
                                toggle_include_date_itself_in_corrected_slices(
                                        self.values["-INCLUDETHRESINCORRS-"])

        def update_date(self,key_prefix,event_handler):
                nums_for_labels = [1,2,4,6] if (self.event.startswith("-GM") or
                                                self.event.startswith("-LM")) else [1]
                current_date = event_handler.get_current_date()
                for i in nums_for_labels:
                        self.window[f'-{key_prefix}{i}CURRENTDATE-'].update(value=current_date)

        def update_plots(self):
                #Subtract one to end date as we want to include it in range
                dates = (list(range(int(self.values["-STARTDATE-"]),int(self.values["-ENDDATE-"])-1,
                                    -int(self.values["-INTERVAL-"]))))
                if self.values["-INCLUDEZEROYBP-"]:
                        dates.append(0)
                self.configuration["dates"] = dates
                self.configuration["sequence_one_base_dir"] = self.values["-SEQUENCEONEVAL-"]
                self.configuration["sequence_two_base_dir"] = self.values["-SEQUENCETWOVAL-"]
                self.configuration["use_latest_version_for_sequence_one"] = self.values["-USELATEST1-"]
                self.configuration["sequence_one_fixed_version"] = self.values["-USEVERSION1-"]
                self.configuration["use_latest_version_for_sequence_two"] = self.values["-USELATEST2-"]
                self.configuration["sequence_two_fixed_version"] = self.values["-USEVERSION2-"]
                self.configuration["super_fine_orography_filepath"] = \
                        self.values["-SUPERFINEOROGFILEPATH-"]
                self.configuration["glacier_mask_file_template"] = \
                        self.values["-GLACMASKTEMPLATE-"]
                for key_prefix in ["GM","LM","CS"]:
                        for i in ([1,2] if key_prefix == "CS" else [1,2,4,6]):
                                self.window[f'-{key_prefix}{i}STARTDATE-'].\
                                        update(value=self.configuration["dates"][0])
                                #The update will reset the current date to the start
                                self.window[f'-{key_prefix}{i}CURRENTDATE-'].\
                                        update(value=self.configuration["dates"][0])
                                self.window[f'-{key_prefix}{i}ENDDATE-'].\
                                        update(value=self.configuration["dates"][-1])
                self.setup_configuration_func(self.configuration,reprocessing=True)

        def reset_to_configuration(self,configuration):
                self.window["-STARTDATE-"].update(value=configuration["dates"][0])
                self.window["-INTERVAL-"].update(value=configuration["dates"][0]
                                                       -configuration["dates"][1])
                self.window["-INCLUDEZEROYBP-"].update(value=(configuration["dates"][-1] == 0))
                if configuration["dates"][-1] == 0:
                        self.window["-ENDDATE-"].update(value=configuration["dates"][-2])
                else:
                        self.window["-ENDDATE-"].update(value=configuration["dates"][-1])
                self.window["-SEQUENCEONEVAL-"].update(value=configuration["sequence_one_base_dir"])
                self.window["-SEQUENCETWOVAL-"].update(value=configuration["sequence_two_base_dir"])
                self.window["-USELATEST1-"].update(value=\
                        configuration["use_latest_version_for_sequence_one"])
                self.window["-USEOTHER1-"].update(value=\
                        not configuration["use_latest_version_for_sequence_one"])
                self.window["-USEVERSION1-"].update(value=\
                        configuration["sequence_one_fixed_version"])
                self.window["-USELATEST2-"].update(value=\
                        configuration["use_latest_version_for_sequence_two"])
                self.window["-USEOTHER2-"].update(value=\
                        not configuration["use_latest_version_for_sequence_two"])
                self.window["-USEVERSION2-"].update(value=\
                        configuration["sequence_two_fixed_version"])
                self.window["-SUPERFINEOROGFILEPATH-"].update(value=\
                        configuration["super_fine_orography_filepath"])
                self.window["-GLACMASKTEMPLATE-"].update(value=\
                        configuration["glacier_mask_file_template"])

        def reset_to_original_config_values(self):
                self.reset_to_configuration(self.initial_configuration)

        def reset_to_current_config_values(self):
                self.reset_to_configuration(self.configuration)

        def set_corrections_file(self,event_handler):
                event_handler.set_corrections_file(self.values["-CORRECTIONSFILE-"])
                self.window['-CURRENTCORRECTIONSFILE-'].\
                        update(value="Current file: {}".format(self.values["-CORRECTIONSFILE-"]))

        def setup_layout(self):
                menu_def = [['File',['Exit',]],['Debug',['Show Plots']]] \
                                if self.dbg_plts is not None else [['File',['Exit',]],]

                input_data_tab_layout = [[sg.Frame("",[[sg.Radio('Lake Agassiz', 'LAKERADIO',default=True)]]),
                                          sg.Frame("",[[sg.Radio('Other lake', 'LAKERADIO'),
                                                        sg.Text("Min lat:"), sg.InputText(size=5), sg.Text("Min lon"), sg.InputText(size=5),
                                                        sg.Text("Max lat:"), sg.InputText(size=5), sg.Text("Max lon"), sg.InputText(size=5)]])],
                                          [sg.Text('Start date: '),
                                           sg.InputText('{}'.format(self.configuration["dates"][0]),
                                                        size=10,key='-STARTDATE-')],
                                          [sg.Text('End date: '),
                                          sg.InputText('{}'.format(self.configuration["dates"][-1]),
                                                       size=10,key='-ENDDATE-')],
                                          [sg.Text('Run interval: '),
                                           sg.InputText('{}'.format(self.configuration["dates"][0] -
                                                                    self.configuration["dates"][1]),size=10,
                                                                    key="-INTERVAL-")],
                                          [sg.Checkbox('Include 0 YBP in list of dates',key="-INCLUDEZEROYBP-")],
                                          [sg.Text('Data source 1:'),
                                           sg.InputText(self.configuration["sequence_one_base_dir"],
                                                        key="-SEQUENCEONEVAL-")],
                                          [sg.Text('Source 1 version'),
                                           sg.Frame("",[[sg.Radio('Use latest version','SOURCERADIO1',
                                                                  default=
                                                                  self.configuration[
                                                                  "use_latest_version_for_sequence_one"],
                                                                  key="-USELATEST1-")]]),
                                           sg.Frame("",[[sg.Radio('Use version:','SOURCERADIO1',
                                                                  default=not
                                                                  self.configuration[
                                                                  "use_latest_version_for_sequence_one"],
                                                                  key="-USEOTHER1-"),
                                                         sg.InputText(self.configuration["sequence_one_fixed_version"],
                                                                      size=5,
                                                                      key="-USEVERSION1-")]])],
                                          [sg.Text('Data source 2:'),sg.InputText(self.configuration["sequence_two_base_dir"],
                                                                                  key="-SEQUENCETWOVAL-")],
                                          [sg.Text('Source 2 version'),
                                           sg.Frame("",[[sg.Radio('Use latest version','SOURCERADIO2',
                                                                  default=
                                                                  self.configuration[
                                                                  "use_latest_version_for_sequence_two"],
                                                                  key="-USELATEST2-")]]),
                                           sg.Frame("",[[sg.Radio('Use version:','SOURCERADIO2',
                                                                  default=not
                                                                  self.configuration[
                                                                  "use_latest_version_for_sequence_two"],
                                                                  key="-USEOTHER2-"),
                                                         sg.InputText(self.configuration["sequence_two_fixed_version"],
                                                                      size=5,
                                                                      key="-USEVERSION2-")]])],
                                          [sg.Text('Super fine orography:'),
                                           sg.InputText(self.configuration["super_fine_orography_filepath"],
                                                        key="-SUPERFINEOROGFILEPATH-")],
                                          [sg.Text('Glacier mask template:'),
                                           sg.InputText(self.configuration["glacier_mask_file_template"],
                                                        key="-GLACMASKTEMPLATE-")],
                                          [sg.Button('Update Plots',key='-UPDATEPLOTS-'),
                                           sg.Button('Return to current values',key='-RETURNTOCURRENT-'),
                                           sg.Button('Reset to default values',key='-RETURNTODEFAULT-')]]

                time_series_tab_layout_wrapper = self.prepare_time_series()

                global_maps_tab_layout_wrapper = self.prepare_maps('GM')

                lake_maps_tab_layout_wrapper = self.prepare_maps('LM')

                cross_sections_tab_layout_wrapper = self.prepare_cross_sections()

                corrections_editor_config_tab_layout = \
                        [[sg.Checkbox("Show corrections editor",key="-SHOWCORREDITOR-",
                                      enable_events=True)],
                         [sg.Text("Current file: {}".format(self.default_corrections_file),
                                  key="-CURRENTCORRECTIONSFILE-")],
                         [sg.Text("Write corrections to file:"),
                          sg.InputText(self.default_corrections_file,
                                       key="-CORRECTIONSFILE-"),
                          sg.Button("Set",
                                    key="-SETCORRECTIONSFILE-",enable_events=True)],
                          [sg.Radio("Corrections for data source 1","CORRSOURCERADIO",
                                    default=True,key="-CORRSOURCERADIO1-",
                                    enable_events=True),
                           sg.Radio("Corrections for data source 2","CORRSOURCERADIO",
                                    default=False,key="-CORRSOURCERADIO2-",
                                    enable_events=True)],
                           [sg.Checkbox("Include threshold date in corrected slices",
                                        default=True,key="-INCLUDETHRESINCORRS-",
                                        enable_events=True)]]

                layout = [[sg.Menu(menu_def)],
                          [sg.TabGroup([[sg.Tab('Input data',input_data_tab_layout,key="-ID-"),
                                         sg.Tab('Time Series',time_series_tab_layout_wrapper,key="-TS-"),
                                         sg.Tab('Global Maps',global_maps_tab_layout_wrapper,key="-GM-"),
                                         sg.Tab('Lake Maps',lake_maps_tab_layout_wrapper,key="-LM-"),
                                         sg.Tab('Cross Sections',cross_sections_tab_layout_wrapper,
                                                key="-CS-"),
                                         sg.Tab('Corrections Editor',corrections_editor_config_tab_layout)]],
                                         key='-TABS-',enable_events=True)]]
                return layout


        def setup_figure(self,figure,canvas_key):
                fig_canvas = FigureCanvasTkAgg(figure,self.window[canvas_key].TKCanvas)
                fig_canvas.draw()
                tbar = NavigationToolbar2Tk(fig_canvas)
                tbar.config(background="#E3F2FD")
                tbar.update()
                fig_canvas.get_tk_widget().pack(fill='both',side='top',expand=1)
                self.fig_canvas[canvas_key] = fig_canvas
                self.tbar_canvas[canvas_key] = tbar


        def run_main_event_loop(self,figures,
                                interactive_timeseries_plots,
                                interactive_plots,
                                interactive_lake_plots,spillway_plots,
                                setup_configuration_func):
                self.interactive_timeseries_plots = interactive_timeseries_plots
                self.interactive_plots = interactive_plots
                self.interactive_lake_plots = interactive_lake_plots
                self.spillway_plots = spillway_plots
                self.setup_configuration_func = setup_configuration_func
                self.window = sg.Window("Paleo Lake Analysis Plots",self.layout,finalize=True,
                                        resizable=True,location=(1852+75,0))
                self.fig_canvas = {}
                self.tbar_canvas = {}
                for key,fig in figures.items():
                        self.setup_figure(fig,key)
                self.interactive_plots.set_corrections_file(self.default_corrections_file)
                self.interactive_lake_plots.set_corrections_file(self.default_corrections_file)
                self.interactive_plots.\
                        set_specify_coords_and_height_callback(SetCoordsAndHeight("GM",
                                                                                  self.window))
                self.interactive_lake_plots.\
                        set_specify_coords_and_height_callback(SetCoordsAndHeight("LM",
                                                                                  self.window))
                while True:
                        self.event, self.values = self.window.read()
                        logging.info(self.event)
                        if self.event in (sg.WIN_CLOSED,'Exit'):
                                break
                        if self.event == 'Show Plots':
                                self.dbg_plts.show_debugging_plots()
                        active_tab = self.values['-TABS-'].strip('-')
                        self.process_config_main_switches_for_time_series()
                        self.process_config_main_switches_for_maps(active_tab)
                        self.process_config_main_switches_for_cross_sections(self.spillway_plots)
                        self.check_for_set_combo_events("GM",self.interactive_plots)
                        self.check_for_set_combo_events("LM",self.interactive_lake_plots)
                        self.check_for_set_combo_events("TS",self.interactive_timeseries_plots)
                        self.check_for_stepping_events("GM",self.interactive_plots)
                        self.check_for_stepping_events("LM",self.interactive_lake_plots)
                        self.check_for_stepping_events("CS",self.spillway_plots)
                        self.check_for_slider_events("GM",self.interactive_plots)
                        self.check_for_slider_events("LM",self.interactive_lake_plots)
                        self.check_for_select_coords_events("GM",self.interactive_plots)
                        self.check_for_select_coords_events("LM",self.interactive_lake_plots)
                        self.check_for_write_correction_events("GM",self.interactive_plots)
                        self.check_for_write_correction_events("LM",self.interactive_lake_plots)
                        self.check_for_correction_source_events([self.interactive_plots,
                                                                 self.interactive_lake_plots])
                        self.check_for_include_threshold_in_corrected_slices_events(
                                [self.interactive_plots,self.interactive_lake_plots])
                        if self.event == "-UPDATEPLOTS-":
                                self.update_plots()
                        if self.event == "-RETURNTODEFAULT-":
                                self.reset_to_original_config_values()
                        if self.event == "-RETURNTOCURRENT-":
                                self.reset_to_current_config_values()
                        if self.event == "-SHOWCORREDITOR-":
                                self.change_correction_editor_visibility()
                        if self.event == "-SETCORRECTIONSFILE-":
                                self.set_corrections_file(self.interactive_plots)
                                self.set_corrections_file(self.interactive_lake_plots)
                        if active_tab in ["GM","LM","CS","TS"]:
                                current_visible_column = self.visible_column[active_tab]
                                if current_visible_column > 0:
                                        key = f'-{active_tab}CANVAS{current_visible_column}-'
                                        self.fig_canvas[key].draw()
                self.window.close()
