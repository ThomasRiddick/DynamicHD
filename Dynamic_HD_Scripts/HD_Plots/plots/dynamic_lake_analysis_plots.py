'''
A module containing plots for analysing the product of dynamic lake
model dry runs

Created on Jun 20, 2021

@author: thomasriddick
'''
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Dynamic_HD_Scripts.base.iodriver import advanced_field_loader
from Dynamic_HD_Scripts.interface.cpp_interface.libs \
    import follow_streams_wrapper
from plotting_utilities.dynamic_lake_analysis_plotting_routines import generate_catchment_and_cflow_comp_sequence
from plotting_utilities.dynamic_lake_analysis_plotting_routines import find_highest_version
from plotting_utilities.dynamic_lake_analysis_plotting_routines import InteractiveTimeslicePlots
from plotting_utilities.color_palette import ColorPalette

global interactive_plots

def rivers_from_lake_corr_and_rivers_from_original_corr_comparison(show_animation=True):
    colors = ColorPalette('default')
    dates = [0]
    dates.extend(list(range(15000,11000,-100)))
    lsmask_sequence = []
    glacier_mask_sequence = []
    catchment_nums_one_sequence = []
    catchment_nums_two_sequence = []
    river_flow_one_sequence = []
    river_flow_two_sequence = []
    river_mouths_one_sequence = []
    river_mouths_two_sequence = []
    default_version = 0
    analysis_base_dir = ("/Users/thomasriddick/Documents/data/"
                         "lake_analysis_runs/lake_analysis_two_26_Mar_2022/")
    use_latest_version = True
    for date in dates:
        if use_latest_version:
            lcorrs_version = \
                    find_highest_version(analysis_base_dir +
                                         "rivers/results/"
                                         "diag_version_VERSION_NUMBER_date_{}".format(date))
        else:
            lcorrs_version = default_version
        if use_latest_version:
            ocorrs_version = \
                    find_highest_version(analysis_base_dir +
                                         "rivers/results/default_orog_corrs/"
                                         "diag_version_VERSION_NUMBER_date_{}".format(date))
        else:
            ocorrs_version = default_version
        lcorrs_results_base_dir = (analysis_base_dir +
                                   "rivers/results/diag_version_{}_date_{}".format(lcorrs_version,date))
        ocorrs_results_base_dir = (analysis_base_dir +
                                   "rivers/results/default_orog_corrs/"
                                   "diag_version_{}_date_{}".format(ocorrs_version,date))
        rdirs = advanced_field_loader(filename=join(ocorrs_results_base_dir,"30min_rdirs.nc"),
                                      time_slice=None,
                                      field_type="RiverDirections",
                                      fieldname="rdirs",
                                      adjust_orientation=True)
        lsmask_data = rdirs.get_lsmask()
        glacier_mask = advanced_field_loader(filename="/Users/thomasriddick/Documents/"
                                             "data/simulation_data/lake_transient_data/run_1/"
                                             "10min_glac_{}k.nc".format(date),
                                             time_slice=None,
                                             fieldname="glac",
                                             adjust_orientation=True)
        catchment_nums_lcorrs = advanced_field_loader(filename=join(lcorrs_results_base_dir,
                                                                    "30min_catchments.nc"),
                                                      time_slice=None,
                                                      fieldname="catchments",
                                                      adjust_orientation=True)
        catchment_nums_ocorrs = advanced_field_loader(filename=join(ocorrs_results_base_dir,
                                                                    "30min_catchments.nc"),
                                                     time_slice=None,
                                                     fieldname="catchments",
                                                     adjust_orientation=True)
        river_flow_lcorrs = advanced_field_loader(filename=join(lcorrs_results_base_dir,
                                                                "30min_flowtocell.nc"),
                                                                time_slice=None,
                                                                fieldname="cumulative_flow",
                                                                adjust_orientation=True)
        river_flow_ocorrs = advanced_field_loader(filename=join(ocorrs_results_base_dir,
                                                                "30min_flowtocell.nc"),
                                                  time_slice=None,
                                                  fieldname="cumulative_flow",
                                                  adjust_orientation=True)
        river_mouths_lcorrs = advanced_field_loader(filename=join(lcorrs_results_base_dir,
                                                                  "30min_flowtorivermouths.nc"),
                                                    time_slice=None,
                                                    fieldname="cumulative_flow_to_ocean",
                                                    adjust_orientation=True)
        river_mouths_ocorrs = advanced_field_loader(filename=join(ocorrs_results_base_dir,
                                                                  "30min_flowtorivermouths.nc"),
                                                    time_slice=None,
                                                    fieldname="cumulative_flow_to_ocean",
                                                    adjust_orientation=True)
        lsmask_sequence.append(lsmask_data)
        glacier_mask_sequence.append(glacier_mask.get_data())
        catchment_nums_one_sequence.append(catchment_nums_lcorrs.get_data())
        catchment_nums_two_sequence.append(catchment_nums_ocorrs.get_data())
        river_flow_one_sequence.append(river_flow_lcorrs.get_data())
        river_flow_two_sequence.append(river_flow_ocorrs.get_data())
        river_mouths_one_sequence.append(river_mouths_lcorrs.get_data())
        river_mouths_two_sequence.append(river_mouths_ocorrs.get_data())
    #For zero slice
    lsmask_zero = [lsmask_sequence[0]]
    glacier_mask_zero = [glacier_mask_sequence[0]],
    catchment_nums_one_zero = [catchment_nums_one_sequence[0]]
    catchment_nums_two_zero = [catchment_nums_two_sequence[0]]
    river_flow_one_zero = [river_flow_one_sequence[0]]
    river_flow_two_zero = [river_flow_two_sequence[0]]
    river_mouths_one_zero = [river_mouths_one_sequence[0]]
    river_mouths_two_zero = [river_mouths_two_sequence[0]]
    if show_animation:
        generate_catchment_and_cflow_comp_sequence(colors,
                                                  lsmask_zero,
                                                  glacier_mask_zero,
                                                  catchment_nums_one_zero,
                                                  catchment_nums_two_zero,
                                                  river_flow_one_zero,
                                                  river_flow_two_zero,
                                                  river_mouths_one_zero,
                                                  river_mouths_two_zero,
                                                  [0],
                                                  minflowcutoff=100,
                                                  use_glacier_mask=False,
                                                  zoomed=False,
                                                  zoomed_section_bounds={})
        fig = plt.figure()
        plt.subplot(111)
        date_text_sequence = [ fig.text(0.4,0.075,"{} YBP".format(date)) for date in dates]
        ims = generate_catchment_and_cflow_comp_sequence(colors,
                                                         lsmask_sequence[1:],
                                                         glacier_mask_sequence[1:],
                                                         catchment_nums_one_sequence[1:],
                                                         catchment_nums_two_sequence[1:],
                                                         river_flow_one_sequence[1:],
                                                         river_flow_two_sequence[1:],
                                                         river_mouths_one_sequence[1:],
                                                         river_mouths_two_sequence[1:],
                                                         date_text_sequence[1:],
                                                         minflowcutoff=100,
                                                         use_glacier_mask=False,
                                                         zoomed=False,
                                                         zoomed_section_bounds={})
        anim = animation.ArtistAnimation(fig,ims,interval=200,blit=False,repeat_delay=500)
        plt.show()
    else:
        interactive_plots = InteractiveTimeslicePlots(colors,
                                                      ["comp"],
                                                      lsmask_sequence[1:],
                                                      glacier_mask_sequence[1:],
                                                      catchment_nums_one_sequence[1:],
                                                      catchment_nums_two_sequence[1:],
                                                      river_flow_one_sequence[1:],
                                                      river_flow_two_sequence[1:],
                                                      river_mouths_one_sequence[1:],
                                                      river_mouths_two_sequence[1:],
                                                      None,None,None,None,None,None,None,
                                                      None,None,None,None,None,None,None,
                                                      dates[1:],
                                                      minflowcutoff=100,
                                                      use_glacier_mask=False,
                                                      zoomed=False,
                                                      zoomed_section_bounds={})

def rivers_from_lake_corr_and_lakes_comparison(show_animation=True):
    colors = ColorPalette('default')
    dates = [0]
    dates.extend(list(range(15990,15890,-10)))
    lsmask_sequence = []
    glacier_mask_sequence = []
    catchment_nums_one_sequence = []
    catchment_nums_two_sequence = []
    river_flow_one_sequence = []
    river_flow_two_sequence = []
    river_mouths_one_sequence = []
    river_mouths_two_sequence = []
    lake_volumes_sequence = []
    default_version = 0
    analysis_base_dir = ("/Users/thomasriddick/Documents/data/"
                         "lake_analysis_runs/lake_analysis_one_21_Jun_2021/")
    use_latest_version = True
    for date in dates:
        if use_latest_version:
            lakes_version = \
                find_highest_version(analysis_base_dir +
                                     "lakes/results/"
                                     "diag_version_VERSION_NUMBER_date_{}".format(date))
        else:
            lakes_version = default_version
        if use_latest_version:
            rivers_version = \
                find_highest_version(analysis_base_dir +
                                     "rivers/results/"
                                     "diag_version_VERSION_NUMBER_date_{}".format(date))
        else:
            rivers_version = default_version
        river_results_base_dir = ( analysis_base_dir +
                                   "rivers/results/diag_version_{}_date_{}".format(lakes_version,date))
        lake_results_base_dir = ( analysis_base_dir +
                                   "lakes/results/diag_version_{}_date_{}".format(rivers_version,date))
        rdirs = advanced_field_loader(filename=join(river_results_base_dir,"30min_rdirs.nc"),
                                      time_slice=None,
                                      field_type="RiverDirections",
                                      fieldname="rdirs",
                                      adjust_orientation=True)
        lsmask_data = rdirs.get_lsmask()
        glacier_mask = advanced_field_loader(filename="/Users/thomasriddick/Documents/"
                                             "data/lake_transient_data/run_1/"
                                             "10min_glac_{}k.nc".format(date),
                                             time_slice=None,
                                             fieldname="glac",
                                             adjust_orientation=True)
        catchment_nums_lake = advanced_field_loader(filename=join(lake_results_base_dir,
                                                                    "30min_connected_catchments.nc"),
                                                      time_slice=None,
                                                      fieldname="catchments",
                                                      adjust_orientation=True)
        catchment_nums_river = advanced_field_loader(filename=join(river_results_base_dir,
                                                                    "30min_catchments.nc"),
                                                     time_slice=None,
                                                     fieldname="catchments",
                                                     adjust_orientation=True)
        river_flow_lake = advanced_field_loader(filename=join(lake_results_base_dir,
                                                                "30min_flowtocell_connected.nc"),
                                                                time_slice=None,
                                                                fieldname="cumulative_flow",
                                                                adjust_orientation=True)
        river_flow_river = advanced_field_loader(filename=join(river_results_base_dir,
                                                                "30min_flowtocell.nc"),
                                                  time_slice=None,
                                                  fieldname="cumulative_flow",
                                                  adjust_orientation=True)
        river_mouths_lake = advanced_field_loader(filename=join(lake_results_base_dir,
                                                                "30min_flowtorivermouths_connected.nc"),
                                                    time_slice=None,
                                                    fieldname="cumulative_flow_to_ocean",
                                                    adjust_orientation=True)
        river_mouths_river = advanced_field_loader(filename=join(river_results_base_dir,
                                                                  "30min_flowtorivermouths.nc"),
                                                    time_slice=None,
                                                    fieldname="cumulative_flow_to_ocean",
                                                    adjust_orientation=True)
        lake_volumes = advanced_field_loader(filename=join(lake_results_base_dir,
                                                           "10min_lake_volumes.nc"),
                                             time_slice=None,
                                             fieldname="lake_volume",
                                             adjust_orientation=True)
        lsmask_sequence.append(lsmask_data)
        glacier_mask_sequence.append(glacier_mask.get_data())
        catchment_nums_one_sequence.append(catchment_nums_lake.get_data())
        catchment_nums_two_sequence.append(catchment_nums_river.get_data())
        river_flow_one_sequence.append(river_flow_lake.get_data())
        river_flow_two_sequence.append(river_flow_river.get_data())
        river_mouths_one_sequence.append(river_mouths_lake.get_data())
        river_mouths_two_sequence.append(river_mouths_river.get_data())
        lake_volumes_sequence.append(lake_volumes.get_data())
    #For zero slice
    lsmask_zero = [lsmask_sequence[0]]
    glacier_mask_zero = [glacier_mask_sequence[0]],
    catchment_nums_one_zero = [catchment_nums_one_sequence[0]]
    catchment_nums_two_zero = [catchment_nums_two_sequence[0]]
    river_flow_one_zero = [river_flow_one_sequence[0]]
    river_flow_two_zero = [river_flow_two_sequence[0]]
    river_mouths_one_zero = [river_mouths_one_sequence[0]]
    river_mouths_two_zero = [river_mouths_two_sequence[0]]
    lake_volumes = [lake_volumes_sequence[0]]
    if show_animation:
        #generate_catchment_and_cflow_comp_sequence(colors,
        #                                           lsmask_zero,
        #                                           glacier_mask_zero,
        #                                           catchment_nums_one_zero,
        #                                           catchment_nums_two_zero,
        #                                           river_flow_one_zero,
        #                                           river_flow_two_zero,
        #                                           river_mouths_one_zero,
        #                                           river_mouths_two_zero,
        #                                           [0]
        #                                           minflowcutoff=100,
        #                                           use_glacier_mask=False,
        #                                           zoomed=False,
        #                                           zoomed_section_bounds={})
        fig = plt.figure()
        date_text_sequence = [ fig.text(0.4,0.075,"{} YBP".format(date)) for date in dates]
        plt.subplot(111)
        ims = generate_catchment_and_cflow_comp_sequence(colors,
                                                         lsmask_sequence[1:],
                                                         glacier_mask_sequence[1:],
                                                         catchment_nums_one_sequence[1:],
                                                         catchment_nums_two_sequence[1:],
                                                         river_flow_one_sequence[1:],
                                                         river_flow_two_sequence[1:],
                                                         river_mouths_one_sequence[1:],
                                                         river_mouths_two_sequence[1:],
                                                         date_text_sequence[1:],
                                                         minflowcutoff=100,
                                                         use_glacier_mask=False,
                                                         zoomed=False,
                                                         zoomed_section_bounds={})
        anim = animation.ArtistAnimation(fig,ims,interval=200,blit=False,repeat_delay=500)
        plt.show()
    else:
        interactive_plots = InteractiveTimeslicePlots(colors,
                                                      ["comp","cflow1","cflow2",
                                                       "catch1","cflowandlake1","lakev1"],
                                                      lsmask_sequence[1:],
                                                      glacier_mask_sequence[1:],
                                                      catchment_nums_one_sequence[1:],
                                                      catchment_nums_two_sequence[1:],
                                                      river_flow_one_sequence[1:],
                                                      river_flow_two_sequence[1:],
                                                      river_mouths_one_sequence[1:],
                                                      river_mouths_two_sequence[1:],
                                                      lake_volumes_sequence[1:],
                                                      None,None,None,None,None,None,
                                                      date_sequence=dates[1:],
                                                      minflowcutoff=100,
                                                      use_glacier_mask=False,
                                                      zoomed=False,
                                                      zoomed_section_bounds={})

def latest_lake_version_vs_base_version_lakes_comparison():
    colors = ColorPalette('default')
    dates = [0]
    dates.extend(list(range(15000,11000,-100)))
    lsmask_sequence = []
    glacier_mask_sequence = []
    catchment_nums_sequence_latest = []
    river_flow_sequence_latest = []
    river_mouths_sequence_latest = []
    lake_volumes_sequence_latest = []
    lake_basin_numbers_sequence_latest = []
    fine_river_flow_sequence_latest = []
    orography_sequence_latest = []
    catchment_nums_sequence_base = []
    river_flow_sequence_base = []
    river_mouths_sequence_base = []
    lake_volumes_sequence_base = []
    lake_basin_numbers_sequence_base = []
    fine_river_flow_sequence_base = []
    orography_sequence_base = []
    analysis_base_dir = ("/Users/thomasriddick/Documents/data/"
                         "lake_analysis_runs/lake_analysis_two_26_Mar_2022/")
    base_version = 0
    for date in dates:
        latest_lakes_version = \
            find_highest_version(analysis_base_dir +
                                 "lakes/results/"
                                 "diag_version_VERSION_NUMBER_date_{}".format(date))
        latest_version_results_base_dir = (analysis_base_dir +
                                           "lakes/results/diag_version_{}_date_{}".\
                                           format(latest_lakes_version,date))
        base_version_results_base_dir = (analysis_base_dir + "lakes/results/diag_version_{}_date_{}".\
                                         format(base_version,date))
        rdirs = advanced_field_loader(filename=join(latest_version_results_base_dir,"30min_rdirs.nc"),
                                      time_slice=None,
                                      field_type="RiverDirections",
                                      fieldname="rdirs",
                                      adjust_orientation=True)
        lsmask_data = rdirs.get_lsmask()
        glacier_mask = advanced_field_loader(filename="/Users/thomasriddick/Documents/"
                                             "data/lake_transient_data/run_1/"
                                             "10min_glac_{}k.nc".format(date),
                                             time_slice=None,
                                             fieldname="glac",
                                             adjust_orientation=True)
        catchment_nums_latest = advanced_field_loader(filename=join(latest_version_results_base_dir,
                                                                    "30min_connected_catchments.nc"),
                                                      time_slice=None,
                                                      fieldname="catchments",
                                                      adjust_orientation=True)
        river_flow_latest = advanced_field_loader(filename=join(latest_version_results_base_dir,
                                                                "30min_flowtocell_connected.nc"),
                                                  time_slice=None,
                                                  fieldname="cumulative_flow",
                                                  adjust_orientation=True)
        river_mouths_latest = advanced_field_loader(filename=join(latest_version_results_base_dir,
                                                                  "30min_flowtorivermouths_connected.nc"),
                                                    time_slice=None,
                                                    fieldname="cumulative_flow_to_ocean",
                                                    adjust_orientation=True)
        lake_volumes_latest = advanced_field_loader(filename=join(latest_version_results_base_dir,
                                                                  "10min_lake_volumes.nc"),
                                                    time_slice=None,
                                                    fieldname="lake_volume",
                                                    adjust_orientation=True)
        lake_basin_numbers_latest = advanced_field_loader(filename=join(latest_version_results_base_dir,
                                                                   "basin_catchment_numbers.nc"),
                                                                   time_slice=None,
                                                                   fieldname="basin_catchment_numbers",
                                                                   adjust_orientation=True)
        # fine_river_flow_latest = advanced_field_loader(filename=join(latest_version_results_base_dir,
        #                                                              "10min_flowtocell.nc"),
        #                                                time_slice=None,
        #                                                fieldname="cumulative_flow",
        #                                                adjust_orientation=True)
        orography_latest = advanced_field_loader(filename=join(latest_version_results_base_dir,
                                                               "10min_corrected_orog.nc"),
                                                 time_slice=None,
                                                 fieldname="corrected_orog",
                                                 adjust_orientation=True)
        catchment_nums_base = advanced_field_loader(filename=join(base_version_results_base_dir,
                                                                  "30min_connected_catchments.nc"),
                                                    time_slice=None,
                                                    fieldname="catchments",
                                                    adjust_orientation=True)
        river_flow_base = advanced_field_loader(filename=join(base_version_results_base_dir,
                                                              "30min_flowtocell_connected.nc"),
                                                time_slice=None,
                                                fieldname="cumulative_flow",
                                                adjust_orientation=True)
        river_mouths_base = advanced_field_loader(filename=join(base_version_results_base_dir,
                                                                "30min_flowtorivermouths_connected.nc"),
                                                  time_slice=None,
                                                  fieldname="cumulative_flow_to_ocean",
                                                  adjust_orientation=True)
        lake_volumes_base = advanced_field_loader(filename=join(base_version_results_base_dir,
                                                                "10min_lake_volumes.nc"),
                                                  time_slice=None,
                                                  fieldname="lake_volume",
                                                  adjust_orientation=True)
        lake_basin_numbers_base = advanced_field_loader(filename=join(base_version_results_base_dir,
                                                                 "basin_catchment_numbers.nc"),
                                                                 time_slice=None,
                                                                 fieldname="basin_catchment_numbers",
                                                                 adjust_orientation=True)
        # fine_river_flow_base = advanced_field_loader(filename=join(base_version_results_base_dir,
        #                                              "10min_flowtocell.nc"),
        #                                              time_slice=None,
        #                                              fieldname="cumulative_flow",
        #                                              adjust_orientation=True)
        orography_base = advanced_field_loader(filename=join(base_version_results_base_dir,
                                                             "10min_corrected_orog.nc"),
                                               time_slice=None,
                                               fieldname="corrected_orog",
                                               adjust_orientation=True)
        lsmask_sequence.append(lsmask_data)
        glacier_mask_sequence.append(glacier_mask.get_data())
        catchment_nums_sequence_latest.append(catchment_nums_latest.get_data())
        river_flow_sequence_latest.append(river_flow_latest.get_data())
        river_mouths_sequence_latest.append(river_mouths_latest.get_data())
        lake_volumes_sequence_latest.append(lake_volumes_latest.get_data())
        lake_basin_numbers_sequence_latest.append(lake_basin_numbers_latest.get_data())
        #fine_river_flow_sequence_latest.append(fine_river_flow_latest.get_data())
        orography_sequence_latest.append(orography_latest.get_data())
        catchment_nums_sequence_base.append(catchment_nums_base.get_data())
        river_flow_sequence_base.append(river_flow_base.get_data())
        river_mouths_sequence_base.append(river_mouths_base.get_data())
        lake_volumes_sequence_base.append(lake_volumes_base.get_data())
        lake_basin_numbers_sequence_base.append(lake_basin_numbers_base.get_data())
        #fine_river_flow_sequence_base.append(fine_river_flow_base.get_data())
        orography_sequence_base.append(orography_base.get_data())
    super_fine_orography = advanced_field_loader(filename=join("/Users/thomasriddick/"
                                                               "Documents/data/HDdata/orographys",
                                                                "srtm30plus_v6.nc"),
                                                 time_slice=None,
                                                 fieldname="topo",
                                                 adjust_orientation=True).get_data()

    first_corrected_orography = advanced_field_loader(filename=join(analysis_base_dir,
                                                                    "corrections","work",
                                                      "pre_preliminary_tweak_orography.nc"),
                                                      time_slice=None,
                                                      fieldname="orog",
                                                      adjust_orientation=True).get_data()
    second_corrected_orography = advanced_field_loader(filename=join(analysis_base_dir,
                                                                     "corrections","work",
                                                       "post_preliminary_tweak_orography.nc"),
                                                       time_slice=None,
                                                       fieldname="orog",
                                                       adjust_orientation=True).get_data()
    third_corrected_orography = advanced_field_loader(filename=join(analysis_base_dir,
                                                                    "corrections","work",
                                                      "pre_final_tweak_orography.nc"),
                                                     time_slice=None,
                                                     fieldname="orog",
                                                     adjust_orientation=True).get_data()
    fourth_corrected_orography = advanced_field_loader(filename=join(analysis_base_dir,
                                                                    "corrections","work",
                                                       "post_final_tweak_orography.nc"),
                                                       time_slice=None,
                                                       fieldname="orog",
                                                       adjust_orientation=True).get_data()
    highest_true_sinks_version = find_highest_version(join(analysis_base_dir,
                                                     "corrections","true_sinks_fields",
                                                     "true_sinks_field_version_"
                                                     "VERSION_NUMBER.nc"))
    true_sinks = advanced_field_loader(filename=join(analysis_base_dir,
                                                     "corrections","true_sinks_fields",
                                                     "true_sinks_field_version_{}.nc".\
                                                     format(highest_true_sinks_version)),
                                                     time_slice=None,
                                                     fieldname="true_sinks",
                                                     adjust_orientation=True).get_data()
    interactive_plots = InteractiveTimeslicePlots(colors,
                                              ["lakev1","cflow1",
                                               "orog1","catch1",
                                               "cflowandlake1",
                                               "firstcorrorog",
                                               "fourthcorrorog",
                                               "lakebasinnums1",
                                               "truesinks"],
                                              lsmask_sequence,
                                              glacier_mask_sequence,
                                              catchment_nums_sequence_latest,
                                              catchment_nums_sequence_base,
                                              river_flow_sequence_latest,
                                              river_flow_sequence_base,
                                              river_mouths_sequence_latest,
                                              river_mouths_sequence_base,
                                              lake_volumes_sequence_latest,
                                              lake_volumes_sequence_base,
                                              lake_basin_numbers_sequence_latest,
                                              lake_basin_numbers_sequence_base,
                                              None,#fine_river_flow_sequence_latest,
                                              None,#fine_river_flow_sequence_base,
                                              orography_sequence_latest,
                                              orography_sequence_base,
                                              super_fine_orography,
                                              first_corrected_orography,
                                              second_corrected_orography,
                                              third_corrected_orography,
                                              fourth_corrected_orography,
                                              true_sinks,
                                              date_sequence=dates,
                                              minflowcutoff=100,
                                              use_glacier_mask=False,
                                              zoomed=False,
                                              zoomed_section_bounds={})

def latest_lake_version_vs_previous_analysis_lakes_comparison():
    colors = ColorPalette('default')
    dates = [0]
    dates.extend(list(range(15000,11000,-500)))
    lsmask_sequence = []
    glacier_mask_sequence = []
    catchment_nums_sequence_latest = []
    river_flow_sequence_latest = []
    river_mouths_sequence_latest = []
    lake_volumes_sequence_latest = []
    lake_basin_numbers_sequence_latest = []
    fine_river_flow_sequence_latest = []
    orography_sequence_latest = []
    catchment_nums_sequence_base = []
    river_flow_sequence_base = []
    river_mouths_sequence_base = []
    lake_volumes_sequence_base = []
    lake_basin_numbers_sequence_base = []
    fine_river_flow_sequence_base = []
    orography_sequence_base = []
    current_analysis_base_dir = ("/Users/thomasriddick/Documents/data/"
                                 "lake_analysis_runs/lake_analysis_two_26_Mar_2022/")
    previous_analysis_base_dir = ("/Users/thomasriddick/Documents/data/"
                                 "lake_analysis_runs/lake_analysis_one_21_Jun_2021/")
    for date in dates:
        latest_lakes_version = \
            find_highest_version(current_analysis_base_dir +
                                 "lakes/results/"
                                 "diag_version_VERSION_NUMBER_date_{}".format(date))
        previous_analysis_lakes_version = 0
            # find_highest_version(previous_analysis_base_dir +
            #                      "lakes/results/"
            #                      "diag_version_VERSION_NUMBER_date_{}".format(date))
        latest_version_results_base_dir = (current_analysis_base_dir +
                                           "lakes/results/diag_version_{}_date_{}".\
                                           format(latest_lakes_version,date))
        previous_analysis_results_base_dir = (previous_analysis_base_dir +
                                              "lakes/results/diag_version_{}_date_{}".\
                                              format(previous_analysis_lakes_version,date))
        rdirs = advanced_field_loader(filename=join(latest_version_results_base_dir,"30min_rdirs.nc"),
                                      time_slice=None,
                                      field_type="RiverDirections",
                                      fieldname="rdirs",
                                      adjust_orientation=True)
        lsmask_data = rdirs.get_lsmask()
        glacier_mask = advanced_field_loader(filename="/Users/thomasriddick/Documents/"
                                             "data/simulation_data/lake_transient_data/run_1/"
                                             "10min_glac_{}k.nc".format(date),
                                             time_slice=None,
                                             fieldname="glac",
                                             adjust_orientation=True)
        catchment_nums_latest = advanced_field_loader(filename=join(latest_version_results_base_dir,
                                                                    "30min_connected_catchments.nc"),
                                                      time_slice=None,
                                                      fieldname="catchments",
                                                      adjust_orientation=True)
        river_flow_latest = advanced_field_loader(filename=join(latest_version_results_base_dir,
                                                                "30min_flowtocell_connected.nc"),
                                                  time_slice=None,
                                                  fieldname="cumulative_flow",
                                                  adjust_orientation=True)
        river_mouths_latest = advanced_field_loader(filename=join(latest_version_results_base_dir,
                                                                  "30min_flowtorivermouths_connected.nc"),
                                                    time_slice=None,
                                                    fieldname="cumulative_flow_to_ocean",
                                                    adjust_orientation=True)
        lake_volumes_latest = advanced_field_loader(filename=join(latest_version_results_base_dir,
                                                                  "10min_lake_volumes.nc"),
                                                    time_slice=None,
                                                    fieldname="lake_volume",
                                                    adjust_orientation=True)
        lake_basin_numbers_latest = advanced_field_loader(filename=join(latest_version_results_base_dir,
                                                                   "basin_catchment_numbers.nc"),
                                                                   time_slice=None,
                                                                   fieldname="basin_catchment_numbers",
                                                                   adjust_orientation=True)
        # fine_river_flow_latest = advanced_field_loader(filename=join(latest_version_results_base_dir,
        #                                                              "10min_flowtocell.nc"),
        #                                                time_slice=None,
        #                                                fieldname="cumulative_flow",
        #                                                adjust_orientation=True)
        orography_latest = advanced_field_loader(filename=join(latest_version_results_base_dir,
                                                               "10min_corrected_orog.nc"),
                                                 time_slice=None,
                                                 fieldname="corrected_orog",
                                                 adjust_orientation=True)
        catchment_nums_base = advanced_field_loader(filename=join(previous_analysis_results_base_dir,
                                                                  "30min_connected_catchments.nc"),
                                                    time_slice=None,
                                                    fieldname="catchments",
                                                    adjust_orientation=True)
        river_flow_base = advanced_field_loader(filename=join(previous_analysis_results_base_dir,
                                                              "30min_flowtocell_connected.nc"),
                                                time_slice=None,
                                                fieldname="cumulative_flow",
                                                adjust_orientation=True)
        river_mouths_base = advanced_field_loader(filename=join(previous_analysis_results_base_dir,
                                                                "30min_flowtorivermouths_connected.nc"),
                                                  time_slice=None,
                                                  fieldname="cumulative_flow_to_ocean",
                                                  adjust_orientation=True)
        lake_volumes_base = advanced_field_loader(filename=join(previous_analysis_results_base_dir,
                                                                "10min_lake_volumes.nc"),
                                                  time_slice=None,
                                                  fieldname="lake_volume",
                                                  adjust_orientation=True)
        lake_basin_numbers_base = advanced_field_loader(filename=join(previous_analysis_results_base_dir,
                                                                 "basin_catchment_numbers.nc"),
                                                                 time_slice=None,
                                                                 fieldname="basin_catchment_numbers",
                                                                 adjust_orientation=True)
        # fine_river_flow_base = advanced_field_loader(filename=join(previous_analysis_results_base_dir,
        #                                              "10min_flowtocell.nc"),
        #                                              time_slice=None,
        #                                              fieldname="cumulative_flow",
        #                                              adjust_orientation=True)
        orography_base = advanced_field_loader(filename=join(previous_analysis_results_base_dir,
                                                             "10min_corrected_orog.nc"),
                                               time_slice=None,
                                               fieldname="corrected_orog",
                                               adjust_orientation=True)
        lsmask_sequence.append(lsmask_data)
        glacier_mask_sequence.append(glacier_mask.get_data())
        catchment_nums_sequence_latest.append(catchment_nums_latest.get_data())
        river_flow_sequence_latest.append(river_flow_latest.get_data())
        river_mouths_sequence_latest.append(river_mouths_latest.get_data())
        lake_volumes_sequence_latest.append(lake_volumes_latest.get_data())
        lake_basin_numbers_sequence_latest.append(lake_basin_numbers_latest.get_data())
        #fine_river_flow_sequence_latest.append(fine_river_flow_latest.get_data())
        orography_sequence_latest.append(orography_latest.get_data())
        catchment_nums_sequence_base.append(catchment_nums_base.get_data())
        river_flow_sequence_base.append(river_flow_base.get_data())
        river_mouths_sequence_base.append(river_mouths_base.get_data())
        lake_volumes_sequence_base.append(lake_volumes_base.get_data())
        lake_basin_numbers_sequence_base.append(lake_basin_numbers_base.get_data())
        #fine_river_flow_sequence_base.append(fine_river_flow_base.get_data())
        orography_sequence_base.append(orography_base.get_data())
    super_fine_orography = advanced_field_loader(filename=join("/Users/thomasriddick/"
                                                               "Documents/data/HDdata/orographys",
                                                                "srtm30plus_v6.nc"),
                                                 time_slice=None,
                                                 fieldname="topo",
                                                 adjust_orientation=True).get_data()

    first_corrected_orography = advanced_field_loader(filename=join(current_analysis_base_dir,
                                                                    "corrections","work",
                                                      "pre_preliminary_tweak_orography.nc"),
                                                      time_slice=None,
                                                      fieldname="orog",
                                                      adjust_orientation=True).get_data()
    second_corrected_orography = advanced_field_loader(filename=join(current_analysis_base_dir,
                                                                     "corrections","work",
                                                       "post_preliminary_tweak_orography.nc"),
                                                       time_slice=None,
                                                       fieldname="orog",
                                                       adjust_orientation=True).get_data()
    third_corrected_orography = advanced_field_loader(filename=join(current_analysis_base_dir,
                                                                    "corrections","work",
                                                      "pre_final_tweak_orography.nc"),
                                                     time_slice=None,
                                                     fieldname="orog",
                                                     adjust_orientation=True).get_data()
    fourth_corrected_orography = advanced_field_loader(filename=join(current_analysis_base_dir,
                                                                    "corrections","work",
                                                       "post_final_tweak_orography.nc"),
                                                       time_slice=None,
                                                       fieldname="orog",
                                                       adjust_orientation=True).get_data()
    highest_true_sinks_version = find_highest_version(join(current_analysis_base_dir,
                                                     "corrections","true_sinks_fields",
                                                     "true_sinks_field_version_"
                                                     "VERSION_NUMBER.nc"))
    true_sinks = advanced_field_loader(filename=join(current_analysis_base_dir,
                                                     "corrections","true_sinks_fields",
                                                     "true_sinks_field_version_{}.nc".\
                                                     format(highest_true_sinks_version)),
                                                     time_slice=None,
                                                     fieldname="true_sinks",
                                                     adjust_orientation=True).get_data()
    interactive_plots = InteractiveTimeslicePlots(colors,
                                                  ["lakev1","cflow1",
                                                   "orog1","catch1",
                                                   "cflowandlake1",
                                                   "firstcorrorog",
                                                   "fourthcorrorog",
                                                   "lakebasinnums1",
                                                   "truesinks"],
                                                  lsmask_sequence,
                                                  glacier_mask_sequence,
                                                  catchment_nums_sequence_latest,
                                                  catchment_nums_sequence_base,
                                                  river_flow_sequence_latest,
                                                  river_flow_sequence_base,
                                                  river_mouths_sequence_latest,
                                                  river_mouths_sequence_base,
                                                  lake_volumes_sequence_latest,
                                                  lake_volumes_sequence_base,
                                                  lake_basin_numbers_sequence_latest,
                                                  lake_basin_numbers_sequence_base,
                                                  None,#fine_river_flow_sequence_latest,
                                                  None,#fine_river_flow_sequence_base,
                                                  orography_sequence_latest,
                                                  orography_sequence_base,
                                                  super_fine_orography,
                                                  first_corrected_orography,
                                                  second_corrected_orography,
                                                  third_corrected_orography,
                                                  fourth_corrected_orography,
                                                  true_sinks,
                                                  date_sequence=dates,
                                                  minflowcutoff=100,
                                                  use_glacier_mask=False,
                                                  zoomed=False,
                                                  zoomed_section_bounds={})


def main():
    #rivers_from_lake_corr_and_rivers_from_original_corr_comparison(False)
    #rivers_from_lake_corr_and_lakes_comparison(False)
    #latest_lake_version_vs_base_version_lakes_comparison()
    latest_lake_version_vs_previous_analysis_lakes_comparison()
    plt.show()

if __name__ == '__main__':
    main()
