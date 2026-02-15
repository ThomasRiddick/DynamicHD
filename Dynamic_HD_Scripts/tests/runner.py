from unittest import TestLoader, TextTestRunner
import unittest
import tests.test_bifurcate_rivers as test_bifurcate_rivers
import tests.test_compute_catchments as test_compute_catchments
import tests.test_connect_coarse_lake_catchments as test_connect_coarse_lake_catchments
import tests.test_cotat_plus_driver as test_cotat_plus_driver
import tests.test_create_connected_lsmask_wrapper as test_create_connected_lsmask_wrapper
import tests.test_determine_river_directions as test_determine_river_directions
import tests.test_dynamic_hd as test_dynamic_hd
import tests.test_fill_sinks_wrapper as test_fill_sinks_wrapper
import tests.test_flow_to_grid_cell as test_flow_to_grid_cell
import tests.test_follow_streams as test_follow_streams
import tests.test_iodriver as test_iodriver
import tests.test_lake_operators as test_lake_operators
import tests.test_loop_breaker_driver as test_loop_breaker_driver
import tests.test_upscale_orography as test_upscale_orography
import tests.test_utilities as test_utilities

import tests.test_dynamic_hd_production_run_driver as test_dynamic_hd_production_run_driver
import tests.test_dynamic_lake_production_run_driver as test_dynamic_lake_production_run_driver

# test_basin_evaluation_algorithm_prototype not included in either set as it
# is only a prototype
short_test_modules = [test_bifurcate_rivers,
                      test_compute_catchments,
                      test_connect_coarse_lake_catchments,
                      test_cotat_plus_driver,
                      test_create_connected_lsmask_wrapper,
                      test_determine_river_directions,
                      test_dynamic_hd,
                      test_fill_sinks_wrapper,
                      test_flow_to_grid_cell,
                      test_follow_streams,
                      test_iodriver,
                      test_lake_operators,
                      test_loop_breaker_driver,
                      test_upscale_orography,
                      test_utilities]

long_test_modules = [test_dynamic_hd_production_run_driver,
                     test_dynamic_lake_production_run_driver]

def create_test_suite(test_modules):
    loader = TestLoader()
    suites = []
    for test_module in test_modules:
        suites.append(loader.loadTestsFromModule(test_module))
    combined_suite = unittest.TestSuite(suites)
    return combined_suite

def create_long_test_suite():
    return create_test_suite(long_test_modules)

def create_short_test_suite():
    return create_test_suite(short_test_modules)

def create_all_test_suite():
    return unittest.TestSuite([create_long_test_suite(),
                               create_short_test_suite()])

if __name__ == "__main__":
    all_test_suite = create_all_test_suite()
    runner = TextTestRunner()
    runner.run(all_test_suite)
