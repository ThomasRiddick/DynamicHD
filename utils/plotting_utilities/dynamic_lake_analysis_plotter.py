'''
Created on Jun 2, 2023

@author: thomasriddick
'''
import sys
import configparser
import logging
from plotting_utilities.dynamic_lake_analysis_plotting_routines import InteractiveTimeSlicePlots
from plotting_utilities.dynamic_lake_analysis_plotting_routines import InteractiveSpillwayPlots
from plotting_utilities.dynamic_lake_analysis_plotting_routines import InteractiveTimeSeriesPlots
from plotting_utilities.dynamic_lake_analysis_plotting_routines import TimeSeriesPlot
from plotting_utilities.dynamic_lake_analysis_plotting_routines import TimeSequence
from plotting_utilities.dynamic_lake_analysis_plotting_routines import TimeSequences
from plotting_utilities.dynamic_lake_analysis_plotting_routines import DataConfiguration
from plotting_utilities.color_palette import ColorPalette
from plotting_utilities import dynamic_lake_analysis_gui as dla_gui
from plotting_utilities.lake_analysis_tools import LakeHeightAndVolumeExtractor
from plotting_utilities.lake_analysis_tools import LakePointExtractor,OutflowBasinIdentifier
from plotting_utilities.lake_analysis_tools import LakeAnalysisDebuggingPlots
from plotting_utilities.lake_analysis_tools import ExitProfiler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logging.basicConfig(level=logging.INFO)

class DynamicLakeAnalysisPlotter:

    lake_defs = {"Agassiz":{"initial_lake_center":(260,500),
                            "lake_emergence_date":14450,
                            "input_area_bounds":{"min_lat":25,
                                                 "max_lat":125,
                                                 "min_lon":125,
                                                 "max_lon":225}}}

    def __init__(self,colors_in,
                 initial_configuration_filepath=None,
                 initial_configuration_in=None,
                 dataset_catalog_filepath=None,
                 debug_plots=False):
        self.colors = colors_in
        if (initial_configuration_filepath is not None and
            initial_configuration_in is not None):
            raise RuntimeError("Can only load a initial configuration from a file or "
                               "pass it in as a dictionary not both")
        elif initial_configuration_filepath is not None:
            self.load_initial_configuration_from_file(initial_configuration_filepath)
        else:
            self.configuration = initial_configuration_in
        self.setup_configuration(self.configuration)
        self.data_configuration = DataConfiguration(dataset_catalog_filepath)
        self.dbg_plts = LakeAnalysisDebuggingPlots() if debug_plots else None

    def load_initial_configuration_from_file(self,initial_configuration_filepath):
        config = configparser.ConfigParser()
        config.read_file(initial_configuration_filepath)
        config_section = config["config"]
        self.configuration = {}
        self.configuration["plots"] = config_section["plots"].split(",")
        start_date = config.getinteger("config","start_date")
        end_date = config.getinteger("config","end_date")
        interval = config.getinteger("config","interval")
        self.configuration["dates"] = list(range(start_date,end_date,-interval))
        self.configuration["sequence_one_base_dir"] = \
            config_section["sequence_one_base_dir"]
        self.configuration["sequence_two_base_dir"] = \
            config_section["sequence_two_base_dir"]
        self.configuration["glacier_mask_file_template"] = \
            config_section["glacier_mask_file_template"]
        self.configuration["super_fine_orography_filepath"] = \
            config_section["super_fine_orography_filepath"]
        self.configuration["use_connected_catchments"] = \
            config.getboolean("config","use_connected_catchments")
        self.configuration["missing_fields"] = config_section["missing_fields"].split(",")
        self.configuration["use_latest_version_for_sequence_one"] = \
            config.getboolean("config","use_latest_version_for_sequence_one")
        self.configuration["sequence_one_fixed_version"] = \
            config.getinteger("config","sequence_one_fixed_version")
        self.configuration["use_latest_version_for_sequence_two"] = \
            config.getboolean("config","use_latest_version_for_sequence_two")
        self.configuration["sequence_two_fixed_version"] = \
            config.getinteger("config","sequence_two_fixed_version")

    def setup_configuration(self,configuration,reprocessing=False):
        self.configuration = configuration
        self.time_sequences = TimeSequences(**self.configuration)
        if reprocessing:
            self.generate_lake_stats()
            self.interactive_plots.replot(self.lake_stats_one["Agassiz"]["lake_points"],
                                          self.lake_stats_two["Agassiz"]["lake_points"],
                                          self.lake_stats_one["Agassiz"]["lake_spillways_masks"],
                                          self.lake_stats_two["Agassiz"]["lake_spillways_masks"],
                                          **vars(self.time_sequences))
            self.interactive_lake_plots.replot(self.lake_stats_one["Agassiz"]["lake_points"],
                                               self.lake_stats_two["Agassiz"]["lake_points"],
                                               self.lake_stats_one["Agassiz"]["lake_spillways_masks"],
                                               self.lake_stats_two["Agassiz"]["lake_spillways_masks"],
                                               **vars(self.time_sequences))
            self.interactive_spillway_plots.replot(lake_center_one_sequence=
                                                   self.lake_stats_one["Agassiz"]["lake_points"],
                                                   lake_center_two_sequence=
                                                   self.lake_stats_two["Agassiz"]["lake_points"],
                                                   lake_potential_spillway_heights_one_sequence=
                                                   self.lake_stats_one["Agassiz"]\
                                                   ["lake_spillway_height_profiles"],
                                                   lake_potential_spillway_heights_two_sequence=
                                                   self.lake_stats_two["Agassiz"]\
                                                   ["lake_spillway_height_profiles"],
                                                   **vars(self.time_sequences))
            self.interactive_timeseries_plots.replot(lake_heights_one_sequence=
                                                     self.lake_stats_one["Agassiz"]["lake_heights"],
                                                     lake_heights_two_sequence=
                                                     self.lake_stats_two["Agassiz"]["lake_heights"],
                                                     lake_volume_one_sequence=
                                                     self.lake_stats_one["Agassiz"]["lake_volumes"],
                                                     lake_volume_two_sequence=
                                                     self.lake_stats_two["Agassiz"]["lake_volumes"],
                                                     lake_outflow_basin_one_sequence=
                                                     self.lake_stats_one["Agassiz"]["lake_outflow_basins"],
                                                     lake_outflow_basin_two_sequence=
                                                     self.lake_stats_two["Agassiz"]["lake_outflow_basins"],
                                                     lake_sill_heights_one_sequence=
                                                     self.lake_stats_one["Agassiz"]["lake_sill_heights"],
                                                     lake_sill_heights_two_sequence=
                                                     self.lake_stats_two["Agassiz"]["lake_sill_heights"],
                                                     **vars(self.time_sequences))

    def generate_lake_stats(self):
        logging.info("Starting lake stats generation")
        self.lake_stats_one = {}
        self.lake_stats_two = {}
        sequences = {"lsmask":
                     self.time_sequences.lsmask_sequence,
                     "connected_lake_basin_numbers_one":
                     self.time_sequences.connected_lake_basin_numbers_one_sequence,
                     "filled_orography_one":
                     self.time_sequences.filled_orography_one_sequence,
                     "lake_volumes_one":
                     self.time_sequences.lake_volumes_one_sequence,
                     "catchment_nums_one":
                     self.time_sequences.catchment_nums_one_sequence,
                     "sinkless_rdirs_one":
                     self.time_sequences.sinkless_rdirs_one_sequence,
                     "orography_one":
                     self.time_sequences.orography_one_sequence,
                     "connected_lake_basin_numbers_two":
                     self.time_sequences.connected_lake_basin_numbers_two_sequence,
                     "filled_orography_two":
                     self.time_sequences.filled_orography_two_sequence,
                     "lake_volumes_two":
                     self.time_sequences.lake_volumes_two_sequence,
                     "catchment_nums_two":
                     self.time_sequences.catchment_nums_two_sequence,
                     "sinkless_rdirs_two":
                     self.time_sequences.sinkless_rdirs_two_sequence,
                     "orography_two":
                     self.time_sequences.orography_two_sequence}
        subsequence_length = 30
        blocks_to_retain = [0,1,2]
        subsequence_collections = [{"date":
            self.time_sequences.date_sequence[i:min(i+subsequence_length,
                                                    len(self.time_sequences.date_sequence))]}
                                              for i in range(0,len(self.time_sequences.date_sequence),
                                                             subsequence_length)]
        for key,sequence in sequences.items():
            for i,subsequence in enumerate(sequence.get_subsequences(subsequence_length)):
                subsequence_collections[i][key] = subsequence
        for subsequence_collection in subsequence_collections:
            for lake_name,lake in self.lake_defs.items():
                self.ocean_basin_identifier.\
                    set_lsmask_sequence(subsequence_collection["lsmask"])
                lake_points = self.lake_point_extractor.\
                    extract_lake_point_sequence(initial_lake_center=lake["initial_lake_center"],
                                                lake_emergence_date=lake["lake_emergence_date"],
                                                dates=subsequence_collection["date"],
                                                input_area_bounds=lake["input_area_bounds"],
                                                connected_lake_basin_numbers_sequence=
                                                subsequence_collection[
                                                "connected_lake_basin_numbers_one"])
                lake_heights,lake_volumes = self.lake_height_and_volume_extractor.\
                    extract_lake_height_and_volume_sequence(lake_point_sequence=lake_points,
                                                            filled_orography_sequence=
                                                            subsequence_collection[
                                                            "filled_orography_one"],
                                                            lake_volumes_sequence=
                                                            subsequence_collection[
                                                            "lake_volumes_one"])
                lake_outflow_basins = self.ocean_basin_identifier.\
                    extract_ocean_basin_for_lake_outflow_sequence(dates=
                                                                  subsequence_collection["date"],
                                                                  input_area_bounds=lake["input_area_bounds"],
                                                                  lake_point_sequence=lake_points,
                                                                  connected_catchments_sequence=
                                                                  subsequence_collection[
                                                                  "catchment_nums_one"],
                                                                  scale_factor=3)
                spillway_height_profiles,spillway_masks = \
                    self.exit_profiler.profile_exit_sequence(lake_center_sequence=lake_points,
                                                             ocean_basin_numbers_sequence=self.ocean_basin_identifier.\
                                                                get_ocean_basin_numbers_sequence(),
                                                            rdirs_sequence=
                                                            subsequence_collection[
                                                            "sinkless_rdirs_one"],
                                                            corrected_heights_sequence=
                                                            subsequence_collection[
                                                            "orography_one"])
                lake_sill_heights = [ [max(profile) for profile in profile_set]
                                      for profile_set in spillway_height_profiles ]
                self.lake_stats_one[lake_name] = {"lake_points":lake_points,
                                                  "lake_heights":lake_heights,
                                                  "lake_volumes":lake_volumes,
                                                  "lake_outflow_basins":lake_outflow_basins,
                                                  "lake_sill_heights":lake_sill_heights,
                                                  "lake_spillway_height_profiles":spillway_height_profiles,
                                                  "lake_spillways_masks":spillway_masks}
                lake_points = self.lake_point_extractor.\
                    extract_lake_point_sequence(initial_lake_center=lake["initial_lake_center"],
                                                lake_emergence_date=lake["lake_emergence_date"],
                                                dates=subsequence_collection["date"],
                                                input_area_bounds=lake["input_area_bounds"],
                                                connected_lake_basin_numbers_sequence=
                                                subsequence_collection[
                                                "connected_lake_basin_numbers_two"])
                lake_heights,lake_volumes = self.lake_height_and_volume_extractor.\
                    extract_lake_height_and_volume_sequence(lake_point_sequence=lake_points,
                                                            filled_orography_sequence=
                                                            subsequence_collection[
                                                            "filled_orography_two"],
                                                            lake_volumes_sequence=
                                                            subsequence_collection[
                                                            "lake_volumes_two"])
                lake_outflow_basins = self.ocean_basin_identifier.\
                    extract_ocean_basin_for_lake_outflow_sequence(dates=
                                                                  subsequence_collection["date"],
                                                                  input_area_bounds=lake["input_area_bounds"],
                                                                  lake_point_sequence=lake_points,
                                                                  connected_catchments_sequence=
                                                                  subsequence_collection[
                                                                  "catchment_nums_two"],
                                                                  scale_factor=3)
                lake_spillway_height_profiles,lake_spillway_masks = \
                    self.exit_profiler.profile_exit_sequence(lake_center_sequence=lake_points,
                                                             ocean_basin_numbers_sequence=self.ocean_basin_identifier.\
                                                                get_ocean_basin_numbers_sequence(),
                                                             rdirs_sequence=
                                                             subsequence_collection[
                                                             "sinkless_rdirs_two"],
                                                             corrected_heights_sequence=
                                                             subsequence_collection[
                                                             "orography_two"])
                lake_sill_heights = [ [max(profile) for profile in profile_set]
                                      for profile_set in spillway_height_profiles ]
                self.lake_stats_two[lake_name] = {"lake_points":lake_points,
                                                  "lake_heights":lake_heights,
                                                  "lake_volumes":lake_volumes,
                                                  "lake_outflow_basins":lake_outflow_basins,
                                                  "lake_sill_heights":lake_sill_heights,
                                                  "lake_spillway_height_profiles":lake_spillway_height_profiles,
                                                  "lake_spillways_masks":lake_spillway_masks}
                for key,sequence in filter(lambda item:type(item[1]) is TimeSequence,sequences.items()):
                    sequence.insert_subsequence_data(subsequence_collection[key])
                    sequence.update_blocks_in_memory()
                    sequence.purge_blocks(blocks_to_retain)
        logging.info("Lake stats generation complete")

    def run(self):
        timeseries = []
        self.corrections = []
        self.lake_height_and_volume_extractor = LakeHeightAndVolumeExtractor()
        self.ocean_basin_identifier = OutflowBasinIdentifier("30minLatLong",self.dbg_plts)
        self.lake_point_extractor = LakePointExtractor()
        self.exit_profiler = ExitProfiler()
        self.generate_lake_stats()
        self.interactive_plots = InteractiveTimeSlicePlots(self.colors,
                                                      self.configuration['plots'],
                                                      **vars(self.time_sequences),
                                                      minflowcutoff=100,
                                                      use_glacier_mask=False,
                                                      dynamic_configuration=True,
                                                      zoomed=False,
                                                      zoomed_section_bounds={},
                                                      lake_points_one=
                                                      self.lake_stats_one["Agassiz"]["lake_points"],
                                                      lake_points_two=
                                                      self.lake_stats_two["Agassiz"]["lake_points"],
                                                      lake_potential_spillway_masks_one=
                                                      self.lake_stats_one["Agassiz"]["lake_spillways_masks"],
                                                      lake_potential_spillway_masks_two=
                                                      self.lake_stats_two["Agassiz"]["lake_spillways_masks"],
                                                      corrections=self.corrections)
        self.interactive_lake_plots = InteractiveTimeSlicePlots(self.colors,
                                                                self.configuration['plots'],
                                                                **vars(self.time_sequences),
                                                                minflowcutoff=100,
                                                                use_glacier_mask=False,
                                                                dynamic_configuration=True,
                                                                zoomed=True,
                                                                zoomed_section_bounds=
                                                                self.lake_defs["Agassiz"]['input_area_bounds'],
                                                                lake_points_one=self.lake_stats_one["Agassiz"]["lake_points"],
                                                                lake_points_two=self.lake_stats_two["Agassiz"]["lake_points"],
                                                                lake_potential_spillway_masks_one=
                                                                self.lake_stats_one["Agassiz"]["lake_spillways_masks"],
                                                                lake_potential_spillway_masks_two=
                                                                self.lake_stats_two["Agassiz"]["lake_spillways_masks"],
                                                                corrections=self.corrections)

        build_dict = lambda figs,nums,prefix : { f'-{prefix}CANVAS{i}-':figure for i,figure in zip(nums,figs) }
        figures = build_dict(self.interactive_plots.figs,[1,2,4,6],'GM')
        #Don't assume python 3.9 and the | syntax
        figures = {**figures,**build_dict(self.interactive_lake_plots.figs,[1,2,4,6],"LM")}
        self.interactive_spillway_plots = \
            InteractiveSpillwayPlots(self.colors,
                                     date_sequence=
                                     self.time_sequences.date_sequence,
                                     lake_center_one_sequence=
                                     self.lake_stats_one["Agassiz"]["lake_points"],
                                     lake_center_two_sequence=
                                     self.lake_stats_two["Agassiz"]["lake_points"],
                                     sinkless_rdirs_one_sequence=
                                     self.time_sequences.sinkless_rdirs_one_sequence,
                                     sinkless_rdirs_two_sequence=
                                     self.time_sequences.sinkless_rdirs_two_sequence,
                                     orography_one_sequence=
                                     self.time_sequences.orography_one_sequence,
                                     orography_two_sequence=
                                     self.time_sequences.orography_two_sequence,
                                     lake_potential_spillway_heights_one_sequence=
                                     self.lake_stats_one["Agassiz"]\
                                     ["lake_spillway_height_profiles"],
                                     lake_potential_spillway_heights_two_sequence=
                                     self.lake_stats_two["Agassiz"]\
                                     ["lake_spillway_height_profiles"],)
        figures = {**figures,**build_dict(self.interactive_spillway_plots.figs,[1,2],"CS")}
        self.interactive_timeseries_plots = \
            InteractiveTimeSeriesPlots(self.colors,
                                       self.data_configuration,
                                       date_sequence=
                                       self.time_sequences.date_sequence,
                                       lake_heights_one_sequence=
                                       self.lake_stats_one["Agassiz"]["lake_heights"],
                                       lake_heights_two_sequence=
                                       self.lake_stats_two["Agassiz"]["lake_heights"],
                                       lake_volume_one_sequence=
                                       self.lake_stats_one["Agassiz"]["lake_volumes"],
                                       lake_volume_two_sequence=
                                       self.lake_stats_two["Agassiz"]["lake_volumes"],
                                       lake_outflow_basin_one_sequence=
                                       self.lake_stats_one["Agassiz"]["lake_outflow_basins"],
                                       lake_outflow_basin_two_sequence=
                                       self.lake_stats_two["Agassiz"]["lake_outflow_basins"],
                                       lake_sill_heights_one_sequence=
                                       self.lake_stats_one["Agassiz"]["lake_sill_heights"],
                                       lake_sill_heights_two_sequence=
                                       self.lake_stats_two["Agassiz"]["lake_sill_heights"],)
        figures = {**figures,**build_dict(self.interactive_timeseries_plots.figs,[1,2,3,4],"TS")}
        gui = dla_gui.DynamicLakeAnalysisGUI(list(self.interactive_plots.plot_types.keys()),
                                             list(self.interactive_timeseries_plots.plot_types.keys()),
                                             self.configuration,self.dbg_plts)
        gui.run_main_event_loop(figures,self.interactive_timeseries_plots,
                                self.interactive_plots,self.interactive_lake_plots,
                                self.interactive_spillway_plots,
                                self.data_configuration,
                                self.setup_configuration,
                                self.poll_io_worker_procs)

    def poll_io_worker_procs(self):
        self.time_sequences.poll_io_worker_procs()

if __name__ == '__main__':
    lake_analysis_plotter = DynamicLakeAnalysisPlotter(ColorPalette('default'),
                                                       initial_configuration_filepath=sys.argv[1])
    lake_analysis_plotter.run()
