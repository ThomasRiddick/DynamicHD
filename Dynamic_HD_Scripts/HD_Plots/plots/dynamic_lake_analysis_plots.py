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
from HD_Plots.utilities.dynamic_lake_analysis_plotting_routines import generate_catchment_and_cflow_comp_sequence
from HD_Plots.utilities.dynamic_lake_analysis_plotting_routines import find_highest_version_for_date
from HD_Plots.utilities.color_palette import ColorPalette

def rivers_from_lake_corr_and_rivers_from_original_corr_comparison():
    colors = ColorPalette('default')
    dates = [0]
    dates.extend(list(range(15990,11000,-10)))
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
                         "lake_analysis_runs/lake_analysis_one_21_Jun_2021/")
    use_latest_version = True
    for date in dates:
        if use_latest_version:
            lcorrs_version = \
                    find_highest_version_for_date(analysis_base_dir +
                                                  "rivers/results/"
                                                  "diag_version_VERSION_NUMBER_date_{}".format(date),date)
        else:
            lcorrs_version = default_version
        if use_latest_version:
            ocorrs_version = \
                    find_highest_version_for_date(analysis_base_dir +
                                                  "rivers/results/default_orog_corrs/"
                                                  "diag_version_VERSION_NUMBER_date_{}".format(date),date)
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
                                             "data/lake_transient_data/run_1/"
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
    #generate_catchment_and_cflow_comp_sequence(colors,
    #                                           lsmask_zero,
    #                                           glacier_mask_zero,
    #                                           catchment_nums_one_zero,
    #                                           catchment_nums_two_zero,
    #                                           river_flow_one_zero,
    #                                           river_flow_two_zero,
    #                                           river_mouths_one_zero,
    #                                           river_mouths_two_zero,
    #                                           [0],
    #                                           minflowcutoff=100,
    #                                           use_glacier_mask=False,
    #                                           zoomed=False,
    #                                           zoomed_section_bounds={})
    fig = plt.figure()
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
                                                     dates,
                                                     minflowcutoff=100,
                                                     use_glacier_mask=False,
                                                     zoomed=False,
                                                     zoomed_section_bounds={})
    anim = animation.ArtistAnimation(fig,ims,interval=200,blit=False,repeat_delay=500)
    plt.show()

def rivers_from_lake_corr_and_lakes_comparison():
    colors = ColorPalette('default')
    dates = [0]
    dates.extend(list(range(15990,10990,-10)))
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
                         "lake_analysis_runs/lake_analysis_one_21_Jun_2021/")
    use_latest_version = True
    for date in dates:
        if use_latest_version:
            lakes_version = \
                find_highest_version_for_date(analysis_base_dir +
                                              "lakes/results/"
                                              "diag_version_VERSION_NUMBER_date_{}".format(date),date)
        else:
            lakes_version = default_version
        if use_latest_version:
            rivers_version = \
                find_highest_version_for_date(analysis_base_dir +
                                              "rivers/results/"
                                              "diag_version_VERSION_NUMBER_date_{}".format(date),date)
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
        lsmask_sequence.append(lsmask_data)
        glacier_mask_sequence.append(glacier_mask.get_data())
        catchment_nums_one_sequence.append(catchment_nums_lake.get_data())
        catchment_nums_two_sequence.append(catchment_nums_river.get_data())
        river_flow_one_sequence.append(river_flow_lake.get_data())
        river_flow_two_sequence.append(river_flow_river.get_data())
        river_mouths_one_sequence.append(river_mouths_lake.get_data())
        river_mouths_two_sequence.append(river_mouths_river.get_data())
    #For zero slice
    lsmask_zero = [lsmask_sequence[0]]
    glacier_mask_zero = [glacier_mask_sequence[0]],
    catchment_nums_one_zero = [catchment_nums_one_sequence[0]]
    catchment_nums_two_zero = [catchment_nums_two_sequence[0]]
    river_flow_one_zero = [river_flow_one_sequence[0]]
    river_flow_two_zero = [river_flow_two_sequence[0]]
    river_mouths_one_zero = [river_mouths_one_sequence[0]]
    river_mouths_two_zero = [river_mouths_two_sequence[0]]
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
                                                     date_text_sequence,
                                                     minflowcutoff=100,
                                                     use_glacier_mask=False,
                                                     zoomed=False,
                                                     zoomed_section_bounds={})
    anim = animation.ArtistAnimation(fig,ims,interval=200,blit=False,repeat_delay=500)
    plt.show()

def main():
    #rivers_from_lake_corr_and_rivers_from_original_corr_comparison()
    rivers_from_lake_corr_and_lakes_comparison()

if __name__ == '__main__':
    main()
