import matplotlib as mpl
import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DynamicLakeAnalysisGUI:

        def __init__(self,avail_plots):
                mpl.use('TkAgg')
                sg.theme("light blue")
                self.avail_plots = avail_plots
                self.layout = self.setup_layout()

        @staticmethod
        def config_and_save_button_factory(key_prefix,key_num):
                return sg.Button('Configure',key=f'-{key_prefix}CONFIGURE{key_num}-'),sg.Button('Save')

        @staticmethod
        def sliders_factory():
                return (sg.Text('Cumulative Flow'),sg.Slider((0,1000),orientation='h'),
                        sg.Text('Minimum orography height'),sg.Slider((0,8000),orientation='h'),
                        sg.Text('Maximum orography height'),sg.Slider((0,8000),orientation='h'),
                        sg.Text("Tip: Click to sides of slider to set precise values"))

        @staticmethod
        def stepping_buttons_factory(key_prefix):
                return (sg.Text('Start date'),
                        sg.Button('E<',key=f"-{key_prefix}EREWIND-",enable_events=True),
                        sg.Button('<<',key=f"-{key_prefix}FREWIND-",enable_events=True),
                        sg.Button('<', key=f"-{key_prefix}REWIND-",enable_events=True),
                        sg.Text("date"),
                        sg.Button('>', key=f"-{key_prefix}FORWARD-",enable_events=True),
                        sg.Button('>>',key=f"-{key_prefix}FFORWARD-",enable_events=True),
                        sg.Button('>E',key=f"-{key_prefix}EFORWARD-",enable_events=True),
                        sg.Text('End date'))

        def prepare_maps(self,key_prefix):
                maps_tab_layout_main_1 = [[*self.config_and_save_button_factory(key_prefix,'1'),
                                           *self.stepping_buttons_factory(f"{key_prefix}1"),
                                           *self.sliders_factory()],
                                          [sg.Canvas(key=f"-{key_prefix}CANVAS1-",
                                                     size=(1800,800))]]

                maps_tab_layout_main_2 =  [[*self.config_and_save_button_factory(key_prefix,'2'),
                                           *self.stepping_buttons_factory(f"{key_prefix}2"),
                                           *self.sliders_factory()],
                                          [sg.Canvas(key=f"-{key_prefix}CANVAS2-",
                                                     size=(1800,800))]]

                maps_tab_layout_main_4 = [[*self.config_and_save_button_factory(key_prefix,'4'),
                                           *self.stepping_buttons_factory(f"{key_prefix}4"),
                                           *self.sliders_factory()],
                                          [sg.Canvas(key=f"-{key_prefix}CANVAS4-",
                                                     size=(1800,800))]]

                maps_tab_layout_main_6 = [[*self.config_and_save_button_factory(key_prefix,'6'),
                                           *self.stepping_buttons_factory(f"{key_prefix}6"),
                                           *self.sliders_factory()],
                                          [sg.Canvas(key=f"-{key_prefix}CANVAS6-",
                                                     size=(1800,800))]]

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
                                                    [sg.Radio('1', f'{key_prefix}PLOTNUMRADIO',key=f'-{key_prefix}PLOTNUMRADIO1-',enable_events=True),
                                                     sg.Radio('2', f'{key_prefix}PLOTNUMRADIO',key=f'-{key_prefix}PLOTNUMRADIO2-',enable_events=True),
                                                     sg.Radio('4', f'{key_prefix}PLOTNUMRADIO',key=f'-{key_prefix}PLOTNUMRADIO4-',default=True,enable_events=True),
                                                     sg.Radio('6', f'{key_prefix}PLOTNUMRADIO',key=f'-{key_prefix}PLOTNUMRADIO6-',enable_events=True)],
                                                    [sg.Column(maps_subpanel_layout_configure_1,key=f'-{key_prefix}LC1-',visible=False),
                                                     sg.Column(maps_subpanel_layout_configure_2,key=f'-{key_prefix}LC2-',visible=False),
                                                     sg.Column(maps_subpanel_layout_configure_4,key=f'-{key_prefix}LC4-',visible=True),
                                                     sg.Column(maps_subpanel_layout_configure_6,key=f'-{key_prefix}LC6-',visible=False)]]

                maps_tab_layout_wrapper = [[sg.Column(maps_tab_layout_main_1,key=f'-{key_prefix}LM1-',visible=False),
                                            sg.Column(maps_tab_layout_main_2,key=f'-{key_prefix}LM2-',visible=False),
                                            sg.Column(maps_tab_layout_main_4,key=f'-{key_prefix}LM4-',visible=True),
                                            sg.Column(maps_tab_layout_main_6,key=f'-{key_prefix}LM6-',visible=False),
                                            sg.Column(maps_tab_layout_configure,key=f'-{key_prefix}LC-',visible=False)]]
                return maps_tab_layout_wrapper

        def change_visible_column(self,key_base,visible_key_num):
                self.window[f'-{key_base}1-'].update(visible=(visible_key_num == 1))
                self.window[f'-{key_base}2-'].update(visible=(visible_key_num == 2))
                self.window[f'-{key_base}4-'].update(visible=(visible_key_num == 4))
                self.window[f'-{key_base}6-'].update(visible=(visible_key_num == 6))

        def process_config_main_switches_for_maps(self,key_prefix):
                if self.event == f"-{key_prefix}BACK-":
                        if self.values[f'-{key_prefix}PLOTNUMRADIO1-']:
                                self.change_visible_column(f'{key_prefix}LM',1)
                        elif self.values[f'-{key_prefix}PLOTNUMRADIO2-']:
                                self.change_visible_column(f'{key_prefix}LM',2)
                        elif self.values[f'-{key_prefix}PLOTNUMRADIO4-']:
                                self.change_visible_column(f'{key_prefix}LM',4)
                        elif self.values[f'-{key_prefix}PLOTNUMRADIO6-']:
                                self.change_visible_column(f'{key_prefix}LM',6)
                        self.window[f'-{key_prefix}LC-'].update(visible=False)
                if (self.event == f'-{key_prefix}CONFIGURE1-' or self.event == f'-{key_prefix}CONFIGURE2-' or
                    self.event == f'-{key_prefix}CONFIGURE4-' or self.event == f'-{key_prefix}CONFIGURE6-'):
                        self.change_visible_column(f'{key_prefix}LM',0)
                        self.window[f'-{key_prefix}LC-'].update(visible=True)
                if self.event == f'-{key_prefix}PLOTNUMRADIO1-':
                        self.change_visible_column(f'{key_prefix}LC',1)
                if self.event == f'-{key_prefix}PLOTNUMRADIO2-':
                        self.change_visible_column(f'{key_prefix}LC',2)
                if self.event == f'-{key_prefix}PLOTNUMRADIO4-':
                        self.change_visible_column(f'{key_prefix}LC',4)
                if self.event == f'-{key_prefix}PLOTNUMRADIO6-':
                        self.change_visible_column(f'{key_prefix}LC',6)

        def check_for_set_combo_events(self,key_prefix):
                combo_events = { f'-{key_prefix}PLOTLIST' + event_label + '-':i for
                                i,event_label in enumerate(["11","21","22",
                                                            "41","42","43","44",
                                                            "61","62","63","64",
                                                            "65","66"])}
                if self.event in combo_events.keys():
                        self.interactive_plots.set_plot_type(combo_events[self.event],
                                                             self.values[self.event])

        def check_for_stepping_events(self,key_prefix):
                forward_stepping_events = [f'-{key_prefix}'+ event_label + 'FORWARD' + '-' for
                                           event_label in ["1","2","4","6"]]
                backward_stepping_events = [f'-{key_prefix}'+ event_label + 'REWIND' + '-' for
                                           event_label in ["1","2","4","6"]]
                if self.event in forward_stepping_events:
                        self.interactive_plots.step_forward()
                if self.event in backward_stepping_events:
                        self.interactive_plots.step_back()



        def setup_layout(self):
                menu_def = [['File',['Exit',]],]

                input_data_tab_layout = [[sg.Frame("",[[sg.Radio('Lake Agassiz', 'LAKERADIO')]]),
                                          sg.Frame("",[[sg.Radio('Other lake', 'LAKERADIO'),
                                                        sg.Text("Min lat:"), sg.InputText(size=5), sg.Text("Min lon"), sg.InputText(size=5),
                                                        sg.Text("Max lat:"), sg.InputText(size=5), sg.Text("Max lon"), sg.InputText(size=5)]])],
                                          [sg.Text('Start date: '),sg.InputText('16000',size=10)],
                                          [sg.Text('End date: '),sg.InputText('11000',size=10)],
                                          [sg.Text('Run interval: '),sg.InputText('100',size=10)],
                                          [sg.Checkbox('Include 0 YBP in list of dates')],
                                          [sg.Text('Data source 1:'),sg.InputText('/path/to/data')],
                                          [sg.Text('Source 1 version'),sg.Frame("",[[sg.Radio('Use latest version','SOURCERADIO1')]]),
                                                                       sg.Frame("",[[sg.Radio('Use version:','SOURCERADIO1'), sg.InputText(size=5)]])],
                                          [sg.Text('Data source 2:'),sg.InputText('/path/to/data')],
                                          [sg.Text('Source 2 version'),sg.Frame("",[[sg.Radio('Use latest version','SOURCERADIO2')]]),
                                           sg.Frame("",[[sg.Radio('Use version:','SOURCERADIO2'), sg.InputText(size=5)]])],
                                          [sg.Text('Super fine orography:'),sg.InputText('/path/to/data')],
                                          [sg.Text('Glacier mask template:'),sg.InputText('/path/to/data')],
                                          [sg.Button('Update Plots'),sg.Button('Return to current values'),sg.Button('Reset to default values')]]

                time_series_tab_layout_configure = [[sg.Button('Back',key='-TSBACK-')],
                                                    [sg.Text("Config")]]

                time_series_tab_layout_main = [[sg.Button('Configure',key='-TSCONFIGURE-'),sg.Button('Save')],
                                               [sg.Canvas(key=f"-LAKEHEIGHT-",
                                                     size=(1800,800))]]

                time_series_tab_layout_wrapper = [[sg.Column(time_series_tab_layout_main,key='-TSLM-'),
                                                   sg.Column(time_series_tab_layout_configure,key='-TSLC-',visible=False)]]

                global_maps_tab_layout_wrapper = self.prepare_maps('GM')

                lake_maps_tab_layout_wrapper = self.prepare_maps('LM')

                cross_sections_tab_layout_configure = [[sg.Button('Back',key='-CSBACK-')],[sg.Text('wiggle')]]

                cross_sections_tab_layout_main = [[sg.Button('Configure',key='-CSCONFIGURE-'),sg.Button('Save'),*self.stepping_buttons_factory(f"CROSS")],
                                                  [sg.Text('Plot')]]

                cross_sections_tab_layout_wrapper = [[sg.Column(cross_sections_tab_layout_main,key='-CSLM-'),
                                                      sg.Column(cross_sections_tab_layout_configure,key='-CSLC-',visible=False)]]

                layout = [[sg.Menu(menu_def)],
                          [sg.TabGroup([[sg.Tab('Input data',input_data_tab_layout),
                                         sg.Tab('Time Series',time_series_tab_layout_wrapper),
                                         sg.Tab('Global Maps',global_maps_tab_layout_wrapper),
                                         sg.Tab('Lake Maps',lake_maps_tab_layout_wrapper),
                                         sg.Tab('Cross Sections',cross_sections_tab_layout_wrapper)]])]]
                return layout


        def setup_figure(self,figure,canvas_key):
                fig_canvas = FigureCanvasTkAgg(figure,self.window[canvas_key].TKCanvas)
                fig_canvas.draw()
                fig_canvas.get_tk_widget().pack(fill='both',side='top',expand=1)

        def run_main_event_loop(self,figures,interactive_plots):
                self.interactive_plots = interactive_plots
                self.window = sg.Window("Paleo Lake Analysis Plots",self.layout,finalize=True,
                                        resizable=True)
                #Need to find a more reliable way to align these list
                for fig,key in zip(figures,
                                   ["-GMCANVAS1-","-GMCANVAS2-",
                                    "-GMCANVAS4-","-GMCANVAS6-",
                                    "-LAKEHEIGHT-"]):
                                    #"-LMCANVAS1-","-LMCANVAS2-",
                                    #"-LMCANVAS4-","-LMCANVAS6-"]):
                        self.setup_figure(fig,key)
                while True:
                        self.event, self.values = self.window.read()
                        if self.event == "-TSBACK-":
                                self.window['-TSLM-'].update(visible=True)
                                self.window['-TSLC-'].update(visible=False)
                        if self.event == "-TSCONFIGURE-":
                                self.window['-TSLM-'].update(visible=False)
                                self.window['-TSLC-'].update(visible=True)
                        self.process_config_main_switches_for_maps('GM')
                        self.process_config_main_switches_for_maps('LM')
                        self.check_for_set_combo_events("GM")
                        self.check_for_set_combo_events("LM")
                        self.check_for_stepping_events("GM")
                        self.check_for_stepping_events("LM")
                        if self.event == "-CSBACK-":
                                self.window['-CSLM-'].update(visible=True)
                                self.window['-CSLC-'].update(visible=False)
                        if self.event == "-CSCONFIGURE-":
                                self.window['-CSLM-'].update(visible=False)
                                self.window['-CSLC-'].update(visible=True)
                        if self.event in (sg.WIN_CLOSED,'Exit'):
                                break
                self.window.close()
