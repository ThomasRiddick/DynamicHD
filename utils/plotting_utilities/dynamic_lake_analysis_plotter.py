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
                            #"lake_emergence_date":14450,
                            "lake_emergence_date":14000, #ice6g
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
        self.configuration["input_orography_file_template_one"] = \
            config_section["input_orography_file_template_one"]
        self.configuration["input_orography_file_template_two"] = \
            config_section["input_orography_file_template_two"]
        self.configuration["present_day_base_input_orography_one_filepath"] = \
            config_section["present_day_base_input_orography_one_filepath"]
        self.configuration["present_day_base_input_orography_two_filepath"] = \
            config_section["present_day_base_input_orography_two_filepath"]
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
        self.configuration["sequence_one_is_transient_run_data"] = \
            config.getboolean("config","sequence_one_is_transient_run_data")
        self.configuration["sequence_two_is_transient_run_data"] = \
            config.getboolean("config","sequence_two_is_transient_run_data")

    def setup_configuration(self,configuration,reprocessing=False):
        self.configuration = configuration
        self.time_sequences = TimeSequences(**self.configuration)
        if reprocessing:
            self.generate_lake_stats()
            self.interactive_plots.replot(self.lake_stats_one["Agassiz"]["lake_points"],
                                          self.lake_stats_two["Agassiz"]["lake_points"],
                                          self.lake_stats_one["Agassiz"]["lake_spillway_masks"],
                                          self.lake_stats_two["Agassiz"]["lake_spillway_masks"],
                                          **vars(self.time_sequences))
            self.interactive_lake_plots.replot(self.lake_stats_one["Agassiz"]["lake_points"],
                                               self.lake_stats_two["Agassiz"]["lake_points"],
                                               self.lake_stats_one["Agassiz"]["lake_spillway_masks"],
                                               self.lake_stats_two["Agassiz"]["lake_spillway_masks"],
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
            self.interactive_timeseries_plots.\
                replot(lake_heights_one_sequence=
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
                       discharge_to_basin_one_sequence=
                       self.lake_stats_one["Agassiz"]["discharge_to_basin"],
                       discharge_to_basin_two_sequence=
                       self.lake_stats_two["Agassiz"]["discharge_to_basin"],
                       filled_lake_volume_one_sequence=
                       self.lake_stats_one["Agassiz"]["filled_lake_volumes"],
                       filled_lake_volume_two_sequence=
                       self.lake_stats_two["Agassiz"]["filled_lake_volumes"],
                       **vars(self.time_sequences))

    def generate_lake_stats(self):
        logging.info("Starting lake stats generation")
        self.lake_stats_one = {}
        self.lake_stats_two = {}
        sequences = {"lsmask_one":
                     self.time_sequences.lsmask_one_sequence,
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
                     "lsmask_two":
                     self.time_sequences.lsmask_two_sequence,
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
        additional_sequences = {"discharge_to_ocean_one":
                                self.time_sequences.discharge_to_ocean_one_sequence,
                                "filled_lake_volumes_one":
                                self.time_sequences.filled_lake_volumes_one_sequence,
                                "discharge_to_ocean_two":
                                self.time_sequences.discharge_to_ocean_two_sequence,
                                "filled_lake_volumes_two":
                                self.time_sequences.filled_lake_volumes_two_sequence}
        for key,value in additional_sequences.items():
            if value is not None:
                sequences[key] = value
        subsequence_length = 5
        blocks_to_retain = [0,1,2]
        subsequence_collections = [{"date":
            self.time_sequences.date_sequence[i:min(i+subsequence_length,
                                                    len(self.time_sequences.date_sequence))]}
                                              for i in range(0,len(self.time_sequences.date_sequence),
                                                             subsequence_length)]
        for key,sequence in sequences.items():
            for i,subsequence in enumerate(sequence.get_subsequences(subsequence_length)):
                subsequence_collections[i][key] = subsequence
        stat_names = ["lake_points","lake_heights","lake_volumes",
                      "lake_outflow_basins","lake_sill_heights",
                      "lake_spillway_height_profiles",
                      "lake_spillway_masks","discharge_to_basin",
                      "filled_lake_volumes"]
        for lake_name in self.lake_defs.keys():
            self.lake_stats_one[lake_name] = {name:[] for name in stat_names}
            self.lake_stats_two[lake_name] = {name:[] for name in stat_names}
        for i,subsequence_collection in enumerate(subsequence_collections):
            for lake_name,lake in self.lake_defs.items():
                for exp in ["one","two"]:
                    unused_stats = []
                    vars(self)[f"ocean_basin_identifier_{exp}"].\
                        set_lsmask_sequence(subsequence_collection[f"lsmask_{exp}"])
                    lake_points = vars(self)[f"lake_point_extractor_{exp}"].\
                        extract_lake_point_sequence(initial_lake_center=lake["initial_lake_center"],
                                                    lake_emergence_date=lake["lake_emergence_date"],
                                                    dates=subsequence_collection["date"],
                                                    connected_lake_basin_numbers_sequence=
                                                    subsequence_collection[
                                                    f"connected_lake_basin_numbers_{exp}"],
                                                    continue_from_previous_subsequence=(i != 0))
                    lake_heights,lake_volumes = self.lake_height_and_volume_extractor.\
                        extract_lake_height_and_volume_sequence(lake_point_sequence=lake_points,
                                                                filled_orography_sequence=
                                                                subsequence_collection[
                                                                f"filled_orography_{exp}"],
                                                                lake_volumes_sequence=
                                                                subsequence_collection[
                                                                f"lake_volumes_{exp}"])
                    if f"filled_lake_volumes_{exp}" in sequences:
                        #Use this only for the filled lake volumes; doesn't apply for
                        #filled lake heights so just insert dummy height data and ignore output
                        _,filled_lake_volumes = self.lake_height_and_volume_extractor.\
                            extract_lake_height_and_volume_sequence(lake_point_sequence=lake_points,
                                                                    filled_orography_sequence=
                                                                    subsequence_collection[
                                                                    f"filled_orography_{exp}"],
                                                                    lake_volumes_sequence=
                                                                    subsequence_collection[
                                                                    f"filled_lake_volumes_{exp}"])
                    else:
                        unused_stats.append("filled_lake_volumes")
                    lake_outflow_basins = vars(self)[f"ocean_basin_identifier_{exp}"].\
                        extract_ocean_basin_for_lake_outflow_sequence(lake_point_sequence=lake_points,
                                                                      connected_catchments_sequence=
                                                                      subsequence_collection[
                                                                      f"catchment_nums_{exp}"],
                                                                      scale_factor=3)
                    if f"discharge_to_ocean_{exp}" in sequences:
                        discharge_to_basin = vars(self)[f"ocean_basin_identifier_{exp}"].\
                            calculate_discharge_to_ocean_basins_sequence(
                                discharge_to_ocean_sequence=subsequence_collection[
                                                            f"discharge_to_ocean_{exp}"])
                    else:
                        unused_stats.append("discharge_to_basin")
                    lake_spillway_height_profiles,lake_spillway_masks = \
                        self.exit_profiler.profile_exit_sequence(lake_center_sequence=lake_points,
                                                                 ocean_basin_numbers_sequence=
                                                                 vars(self)[f"ocean_basin_identifier_{exp}"].\
                                                                    get_ocean_basin_numbers_sequence(),
                                                                rdirs_sequence=
                                                                subsequence_collection[
                                                                f"sinkless_rdirs_{exp}"],
                                                                corrected_heights_sequence=
                                                                subsequence_collection[
                                                                f"orography_{exp}"])
                    lake_sill_heights = [ [max(profile) for profile in profile_set]
                                          for profile_set in lake_spillway_height_profiles ]
                    for stat_name in filter(lambda name:name not in unused_stats ,stat_names):
                        vars(self)[f"lake_stats_{exp}"][lake_name][stat_name].extend(locals()[stat_name])
            for key,sequence in filter(lambda item:type(item[1]) is TimeSequence,sequences.items()):
                sequence.insert_subsequence_data(subsequence_collection[key])
                sequence.update_blocks_in_memory()
                sequence.purge_blocks(blocks_to_retain)
        logging.info("Lake stats generation complete")

    def run(self):
        timeseries = []
        self.corrections = []
        self.lake_height_and_volume_extractor = LakeHeightAndVolumeExtractor()
        self.ocean_basin_identifier_one = OutflowBasinIdentifier("30minLatLong",self.dbg_plts)
        self.ocean_basin_identifier_two = OutflowBasinIdentifier("30minLatLong",self.dbg_plts)
        self.lake_point_extractor_one = LakePointExtractor()
        self.lake_point_extractor_two = LakePointExtractor()
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
                                                      self.lake_stats_one["Agassiz"]["lake_spillway_masks"],
                                                      lake_potential_spillway_masks_two=
                                                      self.lake_stats_two["Agassiz"]["lake_spillway_masks"],
                                                      sequence_one_is_transient_run_data=
                                                      self.configuration[
                                                        "sequence_one_is_transient_run_data"],
                                                      sequence_two_is_transient_run_data=
                                                      self.configuration[
                                                        "sequence_two_is_transient_run_data"],
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
                                                                self.lake_stats_one["Agassiz"]["lake_spillway_masks"],
                                                                lake_potential_spillway_masks_two=
                                                                self.lake_stats_two["Agassiz"]["lake_spillway_masks"],
                                                                sequence_one_is_transient_run_data=
                                                                self.configuration[
                                                                    "sequence_one_is_transient_run_data"],
                                                                sequence_two_is_transient_run_data=
                                                                self.configuration[
                                                                    "sequence_two_is_transient_run_data"],
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
                                       self.lake_stats_two["Agassiz"]["lake_sill_heights"],
                                       discharge_to_basin_one_sequence=
                                       self.lake_stats_one["Agassiz"]["discharge_to_basin"],
                                       discharge_to_basin_two_sequence=
                                       self.lake_stats_two["Agassiz"]["discharge_to_basin"],
                                       filled_lake_volume_one_sequence=
                                       self.lake_stats_one["Agassiz"]["filled_lake_volumes"],
                                       filled_lake_volume_two_sequence=
                                       self.lake_stats_two["Agassiz"]["filled_lake_volumes"])
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
