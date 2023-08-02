'''
Created on Jun 2, 2023

@author: thomasriddick
'''
from plotting_utilities.dynamic_lake_analysis_plotting_routines import InteractiveTimeslicePlots
from plotting_utilities.dynamic_lake_analysis_plotting_routines import TimeSeriesPlot
from plotting_utilities.color_palette import ColorPalette
from plotting_utilities import dynamic_lake_analysis_gui as dla_gui
from plotting_utilities.lake_analysis_tools import LakeHeightAndVolumeExtractor
from plotting_utilities.lake_analysis_tools import LakePointExtractor,OutflowBasinIdentifier
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class DynamicLakeAnalysisPlotter:

    lake_defs = {"Agassiz":{"initial_lake_center":(270,520),
                            "input_area_bounds":{"min_lat":180,
                                                 "max_lat":350,
                                                 "min_lon":440,
                                                 "max_lon":525}}}

    def __init__(self,colors_in,
                 time_sequences_in=None,
                 initial_configuration_in=None):
        self.colors = colors_in
        if time_sequences_in is None:
            self.read_initial_setup()
        else:
            self.time_sequences = time_sequences_in
        if initial_configuration_in is None:
            self.initial_configuration = []
        else:
            self.initial_configuration = initial_configuration_in

    def read_initial_setup():
        raise RuntimeError("not yet implemented")

    def run(self):
        timeseries = []
        interactive_plots = InteractiveTimeslicePlots(self.colors,
                                                      self.initial_configuration,
                                                      **vars(self.time_sequences),
                                                      minflowcutoff=100,
                                                      use_glacier_mask=False,
                                                      zoomed=False,
                                                      zoomed_section_bounds={})
        interactive_spillway_plots = InteractiveSpillwayPlots(self.colors,
                                                              date_sequence=
                                                              self.time_sequences.date_sequence,
                                                              lake_center_one_sequence=
                                                              self.time_sequences.lake_center_one_sequence,
                                                              lake_center_two_sequence=
                                                              self.time_sequences.lake_center_two_sequence,
                                                              sinkless_rdirs_one_sequence=
                                                              self.time_sequences.sinkless_rdirs_one_sequence,
                                                              sinkless_rdirs_two_sequence=
                                                              self.time_sequences.sinkless_rdirs_two_sequence,
                                                              orography_one_sequence=
                                                              self.time_sequences.orography_one_sequence,
                                                              orography_two_sequence=
                                                              self.time_sequences.orography_two_sequence)
        gui = dla_gui.DynamicLakeAnalysisGUI(list(interactive_plots.plot_types.keys()))
        figures = interactive_plots.figs
        fig = plt.figure(figsize=(9,4),facecolor="#E3F2FD",dpi=200)
        figures.append(fig)
        gs=gridspec.GridSpec(nrows=1,ncols=1,width_ratios=[1],
                             height_ratios=[1],
                             hspace=0.1,
                             wspace=0.1)
        timeseries.append(TimeSeriesPlot(fig.add_subplot(gs[0,0])))
        lake_height_and_volume_extractor = LakeHeightAndVolumeExtractor()
        ocean_basin_identifier = OutflowBasinIdentifier("30minLatLong")
        lake_point_extractor = LakePointExtractor()
        lake_stats = {}
        for lake_name,lake in self.lake_defs.items():
            lake_points =  lake_point_extractor.\
                extract_lake_point_sequence(initial_lake_center=lake["initial_lake_center"],
                                            dates=self.time_sequences.date_sequence,
                                            input_area_bounds=lake["input_area_bounds"],
                                            lake_basin_numbers_sequence=
                                            self.time_sequences.lake_basin_numbers_one_sequence)
            lake_heights,lake_volumes = lake_height_and_volume_extractor.\
                extract_lake_height_and_volume_sequence(lake_point_sequence=lake_points,
                                                        filled_orography_sequence=
                                                        self.time_sequences.filled_orography_one_sequence,
                                                        self.time_sequences.lake_volumes_one_sequence)
            lake_outflow_basins = ocean_basin_identifier.\
                extract_ocean_basin_for_lake_outflow_sequence(dates=self.time_sequences.date_sequence,
                                                              input_area_bounds=lake["input_area_bounds"],
                                                              lsmask_sequence=self.time_sequences.lsmask_sequence,
                                                              lake_point_sequence=lake_points,
                                                              connected_catchments_sequence=
                                                              self.time_sequences.catchment_nums_one_sequence)
            lake_stats[lake_name] = {"lake_points":lake_points,
                                     "lake_heights":lake_heights,
                                     "lake_volume":lake_volumes,
                                     "lake_outflow_basins":lake_outflow_basins}
        timeseries[0].ax.plot(self.time_sequences.date_sequence,lake_heights)
        timeseries[0].ax.invert_xaxis()
        gui.run_main_event_loop(figures,interactive_plots)


if __name__ == '__main__':
    lake_analysis_plotter = DynamicLakeAnalysisPlotter(ColorPalette('default'))
    lake_analysis_plotter.run()
j
