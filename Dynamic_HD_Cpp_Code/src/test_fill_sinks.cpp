/*
 * test_fill_sinks.cpp
 *
 * Unit test for the fill sinks C++ code using Google's
 * Google Test Framework
 *
 *  Created on: Mar 17, 2016
 *      Author: thomasriddick
 */

#define GTEST_HAS_TR1_TUPLE 0

#include "gtest/gtest.h"
#include "fill_sinks.hpp"
#include "grid.hpp"
#include "sink_filling_algorithm.hpp"
#include "upscale_orography.hpp"
#include <iostream>
using namespace std;

namespace unittests {

//A global variable equal to the smallest possible double used as a non data value
const double no_data_value = numeric_limits<double>::lowest();

/*
 * Class containing tests of a tarasov-like upscaling technique using algorithm 1 and algorithm 4
 */
class FillSinksOrographyUpscalingTest : public ::testing::Test {

protected:
	bool set_ls_as_no_data_flag = false;
	bool tarasov_mod= true;
	bool debug = false;
	bool* empty_true_sinks_section_five_by_five = new bool[5*5];
	bool* all_land_landsea_mask_five_by_five    = new bool[5*5];
	bool* empty_true_sinks_section_ten_by_ten = new bool[10*10];
	bool* all_land_landsea_mask_ten_by_ten    = new bool[10*10];

public:
	FillSinksOrographyUpscalingTest();
	~FillSinksOrographyUpscalingTest();
};

FillSinksOrographyUpscalingTest::FillSinksOrographyUpscalingTest(){
	for (auto i=0; i < 25; i++) empty_true_sinks_section_five_by_five[i] = false;
	for (auto i=0; i < 25; i++) all_land_landsea_mask_five_by_five[i] = false;
	for (auto i=0; i < 100; i++) empty_true_sinks_section_ten_by_ten[i] = false;
	for (auto i=0; i < 100; i++) all_land_landsea_mask_ten_by_ten[i] = false;
}

FillSinksOrographyUpscalingTest::~FillSinksOrographyUpscalingTest(){

}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellOne){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[5*5] {
		100.0,100.0,81.0,100.0,100.0,
		100.0,100.0,82.0,100.0,100.0,
		100.0,100.0,83.0,100.0,100.0,
		100.0,100.0,84.0,100.0,100.0,
		100.0,100.0,85.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,all_land_landsea_mask_five_by_five,
					  empty_true_sinks_section_five_by_five,
					  new latlon_grid_params(5,5,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,85.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellTwo){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[5*5] {
		100.0,100.0,81.0,100.0,100.0,
		100.0,100.0,82.0,100.0,100.0,
		100.0,100.0,83.0,100.0,100.0,
		100.0,100.0,86.0,100.0,100.0,
		100.0,100.0,85.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,all_land_landsea_mask_five_by_five,
					  empty_true_sinks_section_five_by_five,
					  new latlon_grid_params(5,5,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,86.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellThree){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[5*5] {
		100.0,100.0,81.0,100.0,100.0,
		100.0,100.0,87.0, 82.0, 81.0,
		100.0,100.0,83.0,100.0,100.0,
		100.0,100.0,86.0,100.0,100.0,
		100.0,100.0,85.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,all_land_landsea_mask_five_by_five,
					  empty_true_sinks_section_five_by_five,
					  new latlon_grid_params(5,5,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,82.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellFour){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[5*5] {
		100.0,100.0,81.0,100.0,100.0,
		100.0,100.0,87.0,100.0,100.0,
		100.0,100.0,83.0,100.0,100.0,
		100.0,100.0,86.0,100.0,100.0,
		100.0,100.0,85.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,all_land_landsea_mask_five_by_five,
					  empty_true_sinks_section_five_by_five,
					  new latlon_grid_params(5,5,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,87.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellFive){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0,81.0,100.0,100.0, 100.0,70.0,100.0,100.0,100.0,
		100.0,100.0,87.0,100.0,100.0, 100.0,88.0,100.0,100.0,100.0,
		100.0,100.0,83.0,100.0,100.0, 100.0,89.0,100.0,100.0,100.0,
		100.0,100.0,86.0,100.0,100.0, 100.0,84.0,100.0,100.0,100.0,
		100.0,100.0,85.0,100.0,100.0, 100.0,83.0,100.0,100.0,100.0,
		100.0,100.0,85.0,100.0,100.0, 100.0,82.0,100.0,100.0,100.0,
		100.0,100.0,85.0,100.0,100.0, 100.0,81.0,100.0,100.0,100.0,
		100.0,100.0,85.0,100.0,100.0, 100.0,82.0,100.0,100.0,100.0,
		100.0,84.0,85.0,100.0,100.0,  80.0,100.0,100.0,100.0,100.0,
		83.0,100.0,90.0,100.0,79.0,  100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,87.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellSix){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0,81.0,100.0,100.0, 100.0,70.0,100.0,100.0,100.0,
		100.0,100.0,87.0,100.0,100.0, 100.0,88.0,100.0,100.0,100.0,
		100.0,100.0,83.0,100.0,100.0, 100.0,89.0,100.0,100.0,100.0,
		100.0,100.0,86.0,100.0,100.0, 100.0,84.0,100.0,100.0,100.0,
		100.0,100.0,85.0,100.0,100.0, 100.0,83.0,100.0,100.0,100.0,
		100.0,100.0,85.0,100.0,100.0, 100.0,82.0, 84.0, 84.0, 85.0,
		100.0,100.0,85.0,100.0,100.0, 100.0,81.0,100.0,100.0,100.0,
		100.0,100.0,85.0,100.0,100.0, 100.0,82.0,100.0,100.0,100.0,
		100.0,84.0,85.0,100.0,100.0,  80.0,100.0,100.0,100.0,100.0,
		83.0,100.0,90.0,100.0,79.0,  100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,85.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellSeven){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0, 77.0,100.0,100.0, 100.0,100.0, 92.0,100.0,100.0,
		100.0,100.0, 78.0, 79.0, 79.0,  79.0, 79.0, 80.0,100.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0, 88.0, 82.0,100.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0, 91.0,100.0,100.0,100.0,
		 78.0, 78.0,100.0,100.0,100.0, 100.0, 82.0,100.0,100.0,100.0,
		100.0, 91.5,100.0,100.0,100.0, 100.0, 82.0,100.0,100.0,100.0,
		100.0, 78.0,100.0,100.0,100.0, 100.0, 81.0,100.0,100.0,100.0,
		100.0, 78.0,100.0, 79.0, 79.0,  80.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 78.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,91.5);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellEight){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0, 77.0,100.0,100.0, 100.0,100.0, 92.0,100.0,100.0,
		100.0,100.0, 78.0, 79.0, 79.0,  79.0, 79.0, 80.0,100.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0, 88.0, 82.0,100.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0, 91.0,100.0,100.0,100.0,
		 78.0, 78.0,100.0,100.0,100.0, 100.0, 82.0,100.0,100.0,100.0,
		100.0, 78.0,100.0,100.0,100.0, 100.0, 82.0,100.0,100.0,100.0,
		100.0, 78.0,100.0,100.0,100.0, 100.0, 81.0,100.0,100.0,100.0,
		100.0, 78.0,100.0, 79.0, 79.0,  80.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 78.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,91);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellNine){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* landsea_mask_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false, true,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0, 77.0,100.0,100.0, 100.0,100.0, 92.0,100.0,100.0,
		100.0,100.0, 78.0, 79.0, 79.0,  79.0, 79.0, 80.0,100.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0, 88.0, 82.0,100.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0, 91.0,100.0,100.0,100.0,
		 78.0, 78.0,100.0,100.0,100.0, 100.0, 82.0,100.0,100.0,100.0,
		100.0, 78.0,100.0,100.0,100.0, 100.0, 82.0, 84.0,  0.0,100.0,
		100.0, 78.0,100.0,100.0,100.0, 100.0, 81.0,100.0,100.0,100.0,
		100.0, 78.0,100.0, 79.0, 79.0,  80.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 78.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,84.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellTen){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* landsea_mask_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false, true,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0, 77.0,100.0,100.0, 100.0,100.0, 92.0,100.0,100.0,
		100.0,100.0, 78.0, 79.0, 79.0,  79.0, 79.0, 80.0,100.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0, 88.0, 82.0,100.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0, 91.0,100.0,100.0,100.0,
		 78.0, 78.0,100.0,100.0,100.0, 100.0, 82.0,100.0,100.0,100.0,
		100.0, 78.0,100.0,100.0,100.0, 100.0, 82.0, 84.0, -2.0, -2.0,
		100.0, 78.0,100.0,100.0,100.0, 100.0, 81.0,100.0,100.0,100.0,
		100.0, 78.0,100.0, 79.0, 79.0,  80.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 78.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,84.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellEleven){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* landsea_mask_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false, true,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0, 77.0,100.0,100.0, 100.0,100.0, 92.0,100.0,100.0,
		100.0,100.0, 78.0, 79.0, 79.0,  79.0, 79.0, 80.0,100.0,100.0,
		 78.0, 78.0,100.0,100.0,100.0, 100.0, 88.0, 82.0,100.0,100.0,
		100.0, 78.0,100.0,100.0,100.0, 100.0, 91.0,100.0,100.0,100.0,
		100.0, 78.0,100.0,100.0,100.0, 100.0, 82.0,100.0,100.0,100.0,
		 79.0, 78.0,100.0,100.0,100.0, 100.0, 82.0, 84.0,100.0, -2.0,
		100.0, 78.0,100.0,100.0,100.0, 100.0, 81.0,100.0,100.0,100.0,
		100.0, 78.0,100.0, 79.0, 79.0,  80.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 78.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 75.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,78.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellTwelve){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* landsea_mask_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,true,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0, 71.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0, 70.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0, 70.0, 70.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0, 80.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,71.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellThirteen){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* landsea_mask_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,true,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0, 71.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0, 70.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0, 81.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0, 70.0, 70.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0, 80.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,80.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellFourteen){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* landsea_mask_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,true,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0, 70.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0, 70.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0, 81.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0, 70.0, 71.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0, 80.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,80.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellFifteen){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* landsea_mask_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,true,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 71.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 72.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 73.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 74.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 75.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 76.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,80.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellSixteen){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* landsea_mask_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,true,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, -2.0, 25.0, 51.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,  0.0, 45.0, 46.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,46.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellSeventeen){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* landsea_mask_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,true,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,  0.0, 85.0, 40.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,80.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellEighteen){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* true_sinks_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,true,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,  0.0, 50.0, 55.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  true_sinks_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,55.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellNineteen){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* true_sinks_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,true,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,  0.0, 5.0, -5.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  true_sinks_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,5.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellTwenty){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* true_sinks_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,true,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,  0.0,-5.0, -5.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  true_sinks_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,80.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellTwentyOne){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* true_sinks_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0,   1.0, -5.0,  0.0, -5.0, -5.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  true_sinks_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,80.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellTwentyTwo){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0,81.0,100.0,100.0, 100.0,70.0,100.0,100.0,100.0,
		100.0,100.0,87.0,100.0,100.0, 100.0,88.0,100.0,100.0,100.0,
		100.0,100.0,83.0,100.0,100.0, 100.0,89.0,100.0,100.0,100.0,
		100.0,100.0,86.0,100.0,100.0, 100.0,84.0,100.0,100.0,100.0,
		100.0,100.0,85.0,100.0,100.0, 100.0,83.0,100.0,100.0,100.0,
		100.0,100.0,85.0,100.0,100.0, 100.0,82.0, 84.0, 85.0, 84.0,
		100.0,100.0,85.0,100.0,100.0, 100.0,81.0,100.0,100.0,100.0,
		100.0,100.0,85.0,100.0,100.0, 100.0,82.0,100.0,100.0,100.0,
		100.0,84.0,85.0,100.0,100.0,  80.0,100.0,100.0,100.0,100.0,
		83.0,100.0,90.0,100.0,79.0,  100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,85.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellTwentyThree){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* true_sinks_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 50.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 55.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 60.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 65.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 55.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 50.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  true_sinks_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,80.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellTwentyFour){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* true_sinks_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 50.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 55.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 60.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 65.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 55.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 49.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  true_sinks_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,80.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellTwentyFive){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* true_sinks_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 49.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 55.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 60.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 65.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 55.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 50.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  true_sinks_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,80.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellTwentySix){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* landsea_mask_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false, true,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	bool* true_sinks_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 50.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 55.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 60.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 65.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 55.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 50.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,landsea_mask_ten_by_ten,
					  true_sinks_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,80.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellTwentySeven){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* landsea_mask_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false, true,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	bool* true_sinks_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 49.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 55.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 60.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 65.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 55.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 50.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,landsea_mask_ten_by_ten,
					  true_sinks_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,80.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellTwentyEight){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* landsea_mask_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false, true,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	bool* true_sinks_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 50.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 55.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 60.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 65.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 55.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 49.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,landsea_mask_ten_by_ten,
					  true_sinks_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,80.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellTwentyNine){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* landsea_mask_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false, true,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	bool* true_sinks_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 50.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 55.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 60.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 65.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 55.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 50.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,landsea_mask_ten_by_ten,
					  true_sinks_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,80.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellThirty){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* true_sinks_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 85.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 55.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 50.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 55.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 60.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 65.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 55.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 50.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  true_sinks_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,80.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellThirtyOne){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* true_sinks_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 50.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 55.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 50.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 55.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 60.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 65.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 45.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 70.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  true_sinks_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,80.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellThirtyTwo){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* true_sinks_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 50.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 55.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 50.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 55.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 60.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 65.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 45.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 70.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 75.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 75.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  true_sinks_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,75.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellThirtyThree){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0, 94.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 93.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 92.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 91.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 90.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 89.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 88.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 87.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0, 81.0,
		100.0, 86.0,100.0,100.0,100.0, 100.0,100.0,100.0, 82.0,100.0,
		100.0, 85.0,100.0,100.0,100.0, 100.0,100.0,100.0, 83.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in,1,3.5);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,94.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellThirtyFour){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0, 94.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 93.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 92.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 91.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 90.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 89.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 88.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 87.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0, 81.0,
		100.0, 86.0,100.0,100.0,100.0, 100.0,100.0,100.0, 82.0,100.0,
		100.0, 85.0,100.0,100.0,100.0, 100.0,100.0,100.0, 83.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in,1,3.2);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,83.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellThirtyFive){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0, 94.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 93.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 92.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0, 85.0,
		100.0, 91.0,100.0,100.0,100.0, 100.0,100.0,100.0, 86.0,100.0,
		100.0, 90.0,100.0,100.0,100.0, 100.0,100.0,100.0, 84.0,100.0,
		100.0, 89.0,100.0,100.0,100.0, 100.0,100.0,100.0, 83.0,100.0,
		100.0, 88.0,100.0,100.0,100.0, 100.0,100.0, 82.0,100.0,100.0,
		100.0, 87.0,100.0,100.0,100.0, 100.0,100.0, 83.0,100.0, 81.0,
		100.0, 86.0,100.0,100.0,100.0, 100.0,100.0,100.0, 82.0,100.0,
		100.0, 85.0,100.0,100.0,100.0, 100.0,100.0,100.0, 83.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in,1,3.5);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,86.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellThirtySix){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0, 94.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 93.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 92.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0, 85.0,
		100.0, 91.0,100.0,100.0,100.0, 100.0,100.0,100.0, 86.0,100.0,
		100.0, 90.0,100.0,100.0,100.0, 100.0,100.0,100.0, 84.0,100.0,
		100.0, 89.0,100.0,100.0,100.0, 100.0,100.0,100.0, 83.0,100.0,
		100.0, 88.0,100.0,100.0,100.0, 100.0,100.0, 82.0,100.0,100.0,
		100.0, 87.0,100.0,100.0,100.0, 100.0,100.0, 83.0,100.0, 83.0,
		100.0, 86.0,100.0,100.0,100.0, 100.0,100.0,100.0, 82.0,100.0,
		100.0, 85.0,100.0,100.0,100.0, 100.0,100.0,100.0, 81.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in,1,3.5);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,86.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellThirtySeven){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0, 94.0,100.0,100.0,100.0, 100.0, 88.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0, 87.0,100.0,100.0,100.0,
		100.0, 92.0,100.0,100.0,100.0, 100.0, 86.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0, 85.0,100.0,100.0,100.0,
		100.0, 90.0,100.0,100.0,100.0, 100.0, 84.0,100.0,100.0,100.0,
		100.0,100.0, 89.0,100.0,100.0, 100.0, 83.0,100.0,100.0,100.0,
		100.0, 88.0,100.0,100.0,100.0, 100.0,100.0, 82.0,100.0,100.0,
		100.0,100.0, 87.0,100.0,100.0, 100.0,100.0, 83.0,100.0,100.0,
		100.0, 86.0,100.0,100.0,100.0, 100.0,100.0,100.0, 82.0,100.0,
		100.0,100.0, 85.0,100.0,100.0, 100.0,100.0,100.0, 81.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in,1,10.8);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,88.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellThirtyEight){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0, 94.0,100.0,100.0,100.0, 100.0, 88.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0, 87.0,100.0,100.0,100.0,
		100.0, 92.0,100.0,100.0,100.0, 100.0, 86.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0, 85.0,100.0,100.0,100.0,
		100.0, 90.0,100.0,100.0,100.0, 100.0, 84.0,100.0,100.0,100.0,
		100.0,100.0, 89.0,100.0,100.0, 100.0, 83.0,100.0,100.0,100.0,
		100.0, 88.0,100.0,100.0,100.0, 100.0,100.0, 82.0,100.0,100.0,
		100.0,100.0, 87.0,100.0,100.0, 100.0,100.0, 83.0,100.0,100.0,
		100.0, 86.0,100.0,100.0,100.0, 100.0,100.0,100.0, 82.0,100.0,
		100.0,100.0, 85.0,100.0,100.0, 100.0,100.0,100.0, 81.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in,1,11.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,94.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellThirtyNine){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0, 94.0,100.0,100.0,100.0, 100.0, 88.0,100.0,100.0,100.0,
		100.0, 93.0,100.0,100.0,100.0, 100.0, 87.0,100.0,100.0,100.0,
		100.0, 92.0,100.0,100.0,100.0, 100.0, 86.0,100.0,100.0,100.0,
		100.0, 91.0,100.0,100.0,100.0, 100.0, 85.0,100.0,100.0,100.0,
		100.0, 90.0,100.0,100.0,100.0, 100.0, 84.0,100.0,100.0,100.0,
		100.0, 89.0,100.0,100.0,100.0, 100.0, 83.0,100.0,100.0,100.0,
		100.0, 88.0,100.0,100.0,100.0, 100.0,100.0, 82.0,100.0,100.0,
		100.0, 87.0,100.0,100.0,100.0,  85.0, 84.0, 83.0,100.0,100.0,
		100.0, 86.0,100.0,100.0,100.0,  86.0,100.0,100.0, 82.0,100.0,
		100.0, 85.0,100.0,100.0,100.0,  87.0,100.0,100.0, 81.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in,1,10.8);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,88.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellForty){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0, 94.0,100.0,100.0,100.0, 100.0, 88.0,100.0,100.0,100.0,
		100.0, 93.0,100.0,100.0,100.0, 100.0, 87.0,100.0,100.0,100.0,
		100.0, 92.0,100.0,100.0,100.0, 100.0, 86.0,100.0,100.0,100.0,
		100.0, 91.0,100.0,100.0,100.0, 100.0, 85.0,100.0,100.0,100.0,
		100.0, 90.0,100.0,100.0,100.0, 100.0, 84.0,100.0,100.0,100.0,
		100.0, 89.0,100.0,100.0,100.0, 100.0, 83.0,100.0,100.0,100.0,
		100.0, 88.0,100.0,100.0,100.0, 100.0,100.0, 82.0,100.0,100.0,
		100.0, 87.0,100.0,100.0,100.0,  85.0, 84.0, 83.0,100.0,100.0,
		100.0, 86.0,100.0,100.0,100.0,  86.0,100.0,100.0, 82.0,100.0,
		100.0, 85.0,100.0,100.0,100.0,  87.0,100.0,100.0, 81.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in,1,6.8);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,87.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellFortyOne){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0, 94.0,100.0,100.0,100.0, 100.0, 88.0,100.0,100.0,100.0,
		100.0, 93.0,100.0,100.0,100.0, 100.0, 87.0,100.0,100.0,100.0,
		100.0, 92.0,100.0,100.0,100.0, 100.0, 86.0,100.0,100.0,100.0,
		100.0, 91.0,100.0,100.0,100.0, 100.0, 85.0,100.0,100.0,100.0,
		100.0, 90.0,100.0,100.0,100.0, 100.0, 84.0,100.0,100.0,100.0,
		100.0, 89.0,100.0,100.0,100.0, 100.0, 83.0,100.0,100.0,100.0,
		100.0, 88.0,100.0,100.0,100.0, 100.0,100.0, 82.0,100.0,100.0,
		100.0, 87.0,100.0,100.0,100.0,  85.0, 84.0, 83.0,100.0,100.0,
		100.0, 86.0,100.0,100.0,100.0,  86.0,100.0,100.0, 82.0,100.0,
		100.0, 85.0,100.0,100.0,100.0,  87.0,100.0,100.0, 81.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in,1,7.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,88.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellFortyTwo){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 59.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0,100.0,100.0, 53.0,100.0, 100.0, 92.0, 91.0,100.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0, 93.0,100.0,100.0,100.0,
		100.0, 51.0,100.0,100.0,100.0, 100.0,100.0, 93.0, 94.0,100.0,
		 50.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		100.0, 51.0, 52.0, 53.0, 54.0, 100.0,100.0, 95.0,100.0,100.0,
		100.0, 52.0,100.0,100.0, 55.0, 100.0,100.0, 97.0, 96.0,100.0,
		100.0, 53.0,100.0,100.0, 56.0, 100.0, 98.0,100.0,100.0,100.0,
		100.0, 54.0,100.0,100.0, 57.0, 100.0, 98.0,100.0,100.0,100.0,
		100.0, 58.0,100.0,100.0, 60.0, 100.0,100.0, 99.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in,1,6.4);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,58.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellFortyThree){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 59.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0,100.0,100.0, 53.0,100.0, 100.0, 92.0, 91.0,100.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0, 93.0,100.0,100.0,100.0,
		100.0, 51.0,100.0,100.0,100.0, 100.0,100.0, 93.0, 94.0,100.0,
		 50.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		100.0, 51.0, 52.0, 53.0, 54.0, 100.0,100.0, 95.0,100.0,100.0,
		100.0, 52.0,100.0,100.0, 55.0, 100.0,100.0, 97.0, 96.0,100.0,
		100.0, 53.0,100.0,100.0, 56.0, 100.0, 98.0,100.0,100.0,100.0,
		100.0, 54.0,100.0,100.0, 57.0, 100.0, 98.0,100.0,100.0,100.0,
		100.0, 58.0,100.0,100.0, 60.0, 100.0,100.0, 99.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in,1,6.6);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,59.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellFortyFour){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 59.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0,100.0,100.0, 53.0,100.0, 100.0, 92.0, 91.0,100.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0, 93.0,100.0,100.0,100.0,
		100.0, 51.0,100.0,100.0,100.0, 100.0,100.0, 93.0, 94.0,100.0,
		 50.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		100.0, 51.0, 52.0, 53.0, 54.0, 100.0,100.0, 95.0,100.0,100.0,
		100.0, 52.0,100.0,100.0, 55.0, 100.0,100.0, 97.0, 96.0,100.0,
		100.0, 53.0,100.0,100.0, 56.0, 100.0, 98.0,100.0,100.0,100.0,
		100.0, 54.0,100.0,100.0, 57.0, 100.0, 98.0,100.0,100.0,100.0,
		100.0, 58.0,100.0,100.0, 60.0, 100.0,100.0, 99.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in,1,7.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,60.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellFortyFive){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 59.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0,100.0,100.0, 53.0,100.0, 100.0, 92.0, 91.0,100.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0, 93.0,100.0,100.0,100.0,
		100.0, 51.0,100.0,100.0,100.0, 100.0,100.0, 93.0, 94.0,100.0,
		 50.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		 51.0,100.0, 52.0, 53.0, 54.0, 100.0,100.0, 95.0,100.0,100.0,
		 52.0,100.0,100.0,100.0, 55.0, 100.0,100.0, 97.0, 96.0,100.0,
		 53.0,100.0,100.0,100.0, 56.0, 100.0, 98.0,100.0,100.0,100.0,
		 52.0,100.0,100.0,100.0, 57.0, 100.0, 98.0,100.0,100.0,100.0,
		 58.0,100.0,100.0,100.0, 60.0, 100.0,100.0, 99.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in,1,6.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,58.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellFortySix){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 59.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0,100.0,100.0, 53.0,100.0, 100.0, 92.0, 91.0,100.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0, 93.0,100.0,100.0,100.0,
		100.0, 51.0,100.0,100.0,100.0, 100.0,100.0, 93.0, 94.0,100.0,
		 50.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		 51.0,100.0, 52.0, 53.0, 54.0, 100.0,100.0, 95.0,100.0,100.0,
		 52.0,100.0,100.0,100.0, 55.0, 100.0,100.0, 97.0, 96.0,100.0,
		 53.0,100.0,100.0,100.0, 56.0, 100.0, 98.0,100.0,100.0,100.0,
		 52.0,100.0,100.0,100.0, 57.0, 100.0, 98.0,100.0,100.0,100.0,
		 58.0,100.0,100.0,100.0, 60.0, 100.0,100.0, 99.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in,1,6.1);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,59.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellFortySeven){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 59.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0,100.0,100.0, 53.0,100.0, 100.0, 92.0, 91.0,100.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0, 93.0,100.0,100.0,100.0,
		100.0, 51.0,100.0,100.0,100.0, 100.0,100.0, 93.0, 94.0,100.0,
		 50.0,100.0, 51.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		 51.0,100.0, 52.0, 53.0, 54.0, 100.0,100.0, 95.0,100.0,100.0,
		 52.0,100.0,100.0,100.0, 55.0, 100.0,100.0, 97.0, 96.0,100.0,
		 53.0,100.0,100.0,100.0, 56.0, 100.0, 98.0,100.0,100.0,100.0,
		 52.0,100.0,100.0,100.0, 57.0, 100.0, 98.0,100.0,100.0,100.0,
		 54.0, 55.0, 56.0,100.0, 60.0, 100.0,100.0, 99.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in,1,7.4);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,56.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellFortyEight){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0, 59.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0,100.0,100.0, 53.0,100.0, 100.0, 92.0, 91.0,100.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0, 93.0,100.0,100.0,100.0,
		100.0, 51.0,100.0,100.0,100.0, 100.0,100.0, 93.0, 94.0,100.0,
		 50.0,100.0, 51.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		 51.0,100.0, 52.0, 53.0, 54.0, 100.0,100.0, 95.0,100.0,100.0,
		 52.0,100.0,100.0,100.0, 55.0, 100.0,100.0, 97.0, 96.0,100.0,
		 53.0,100.0,100.0,100.0, 56.0, 100.0, 98.0,100.0,100.0,100.0,
		 52.0,100.0,100.0,100.0, 57.0, 100.0, 98.0,100.0,100.0,100.0,
		 54.0, 55.0, 56.0,100.0, 60.0, 100.0,100.0, 99.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in,1,7.5);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,60.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellFortyNine){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0, 78.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0, 77.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0, 76.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0, 75.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in,1,5.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellFifty){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0, 78.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0, 77.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0, 76.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0, 75.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in,1,4.8);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,78.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellFortyFiftyOne){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0, 78.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0, 79.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0, 80.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0, 81.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in,1,5.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellFiftyTwo){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0, 78.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0, 79.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0, 80.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0, 81.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in,1,4.8);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,81.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellFiftyThree){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* landsea_mask_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false, true,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0, 52.0, 51.0,50.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,1,3.1);
	alg1.setup_fields(orography_section,landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellFiftyFour){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* landsea_mask_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false, true,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0, 52.0, 51.0, 50.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,1,2.9);
	alg1.setup_fields(orography_section,landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,52.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellFiftyFive){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* landsea_mask_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false, true,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0, 50.0, 51.0, 52.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,1,3.1);
	alg1.setup_fields(orography_section,landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellFiftySix){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* landsea_mask_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false, true,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0, 50.0, 51.0, 52.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,1,2.9);
	alg1.setup_fields(orography_section,landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,52.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellFiftySeven){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* true_sinks_section_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false, true,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0, 51.0, 52.0, 51.0,50.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,1,4.1);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellFiftyEight){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* true_sinks_section_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false, true,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0, 51.0, 52.0, 51.0, 50.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,1,3.9);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,52.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellFiftyNine){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* true_sinks_section_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false, true,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0, 49.0, 50.0, 51.0, 52.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,1,4.1);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellSixty){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* true_sinks_section_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false, true,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0, 49.0, 50.0, 51.0, 52.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,1,3.9);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,52.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellSixtyOne){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* true_sinks_section_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0, 55.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0, 54.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0, 53.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0, 52.0, 51.0, 50.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,1,6.3);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,55.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellSixtyTwo){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* true_sinks_section_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0, 55.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0, 54.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0, 53.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0, 52.0, 51.0, 50.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,1,6.5);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellSixtyThree){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* true_sinks_section_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false, true,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0, 55.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0, 54.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0, 53.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0, 52.0, 51.0, 50.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,1,6.5);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellSixtyFour){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0, 52.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0, 51.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0, 50.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,1,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,52.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellSixtyFive){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0, 52.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0, 51.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0, 50.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,2,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellSixtySix){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0,  51.0,100.0, 65.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0,  52.0,100.0, 64.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0,  53.0,100.0, 63.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0,  54.0,100.0, 62.0,100.0,100.0,
		100.0,100.0, 94.0,100.0,100.0,  55.0,100.0, 61.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0,  56.0,100.0, 60.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0,  57.0,100.0, 59.0,100.0,100.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0, 58.0,100.0,100.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,7,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,65.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellSixtySeven){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0,  51.0,100.0, 65.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0,  52.0,100.0, 64.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0,  53.0,100.0, 63.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0,  54.0,100.0, 62.0,100.0,100.0,
		100.0,100.0, 94.0,100.0,100.0,  55.0,100.0, 61.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0,  56.0,100.0, 60.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0,  57.0,100.0, 59.0,100.0,100.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0, 58.0,100.0,100.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,8,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellSixtyEight){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0, 52.0, 53.0,100.0,100.0,
		100.0,100.0, 98.0,100.0,100.0,  51.0,100.0,100.0, 54.0,100.0,
		100.0,100.0, 99.0,100.0,100.0,  50.0,100.0,100.0, 55.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,2,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,55.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellSixtyNine){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0, 52.0, 53.0,100.0,100.0,
		100.0,100.0, 98.0,100.0,100.0,  51.0,100.0,100.0, 54.0,100.0,
		100.0,100.0, 99.0,100.0,100.0,  50.0,100.0,100.0, 55.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellSeventy){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		 52.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 51.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		 50.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,1,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,52.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellSeventyOne){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		 52.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 51.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		 50.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,2,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,94.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellSeventyTwo){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0, 55.0, 56.0, 57.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0, 54.0,100.0,100.0,100.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0, 53.0,100.0,100.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0, 52.0,100.0,100.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0, 51.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,5,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,57.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellSeventyThree){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0, 55.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0, 54.0,100.0, 56.0,100.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0, 53.0,100.0, 57.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0, 52.0,100.0, 57.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0, 51.0,100.0, 57.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,5,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellSeventyFour){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0, 50.0,100.0, 54.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0, 51.0,100.0, 53.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0, 52.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellSeventyFive){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0, 50.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0, 51.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0, 52.0, 53.0, 54.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,54.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellSeventySix){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0, 53.0, 54.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0, 52.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0,100.0, 51.0, 50.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellSeventySeven){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0,100.0, 54.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0,100.0, 53.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0, 52.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0,100.0, 51.0, 50.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,54.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellSeventyEight){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 91.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 92.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 93.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		 50.0, 51.0,100.0,100.0,100.0, 100.0,100.0,100.0, 95.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0,100.0,100.0, 96.0,100.0,
		100.0,100.0, 53.0,100.0,100.0, 100.0,100.0,100.0, 97.0,100.0,
		100.0,100.0, 54.0,100.0,100.0, 100.0,100.0,100.0, 98.0,100.0,
		100.0,100.0, 55.0,100.0,100.0, 100.0,100.0,100.0, 99.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,55.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellSeventyNine){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 91.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 92.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 93.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		 50.0, 51.0,100.0,100.0,100.0, 100.0,100.0,100.0, 95.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0,100.0,100.0, 96.0,100.0,
		100.0,100.0, 53.0,100.0,100.0, 100.0,100.0,100.0, 97.0,100.0,
		 55.0, 54.0,100.0,100.0,100.0, 100.0,100.0,100.0, 98.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 99.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellEighty){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0, 58.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0, 54.0,100.0, 56.0,100.0,
		100.0,100.0, 97.0,100.0,100.0,  54.0, 53.0,100.0, 57.0,100.0,
		100.0,100.0, 98.0,100.0, 55.0, 100.0, 52.0,100.0, 57.0,100.0,
		100.0,100.0, 99.0,100.0, 56.0, 100.0, 51.0,100.0, 57.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,5,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellEightyOne){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0, 58.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0, 54.0,100.0, 56.0,100.0,
		100.0,100.0, 97.0,100.0,100.0,  54.0, 53.0,100.0, 57.0,100.0,
		100.0,100.0, 98.0,100.0, 55.0, 100.0, 52.0,100.0, 57.0,100.0,
		100.0,100.0, 99.0,100.0, 56.0, 100.0, 51.0,100.0, 57.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,4,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,58.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellEightyTwo){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0, 58.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0, 54.0,100.0, 56.0,100.0,
		100.0,100.0, 97.0,100.0,100.0,  54.0, 53.0,100.0, 57.0,100.0,
		100.0,100.0, 98.0,100.0, 55.0, 100.0, 52.0,100.0, 57.0,100.0,
		100.0,100.0, 99.0,100.0, 56.0, 100.0, 51.0,100.0, 57.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,2,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,56.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellEightyThree){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* landsea_mask_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false, true,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false, true,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 91.0,100.0,
		100.0,100.0, 55.0,100.0,100.0, 100.0,100.0,100.0, 92.0,100.0,
		100.0,100.0, 53.0,100.0,100.0, 100.0,100.0,100.0, 93.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		 50.0, 51.0,100.0,100.0,100.0, 100.0,100.0,100.0, 95.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 96.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 97.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 98.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 99.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0);
	alg1.setup_fields(orography_section,landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,55.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellEightyFour){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* landsea_mask_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false, true,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false, true,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 91.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 92.0,100.0,
		 55.0, 53.0,100.0,100.0,100.0, 100.0,100.0,100.0, 93.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		 50.0, 51.0,100.0,100.0,100.0, 100.0,100.0,100.0, 95.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 96.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 97.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 98.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 99.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0);
	alg1.setup_fields(orography_section,landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellEightyFive){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* true_sinks_section_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false, true,false,false,false,false,false,false,false,
		false,false,false,false,false,false, true,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 91.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0,100.0,100.0, 92.0,100.0,
		100.0,100.0, 53.0,100.0,100.0, 100.0,100.0,100.0, 93.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		 50.0, 51.0,100.0,100.0,100.0, 100.0,100.0,100.0, 95.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 96.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 97.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 98.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 99.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,53.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellEightySix){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* true_sinks_section_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false, true,false,false,false,false,false,false,false,
		false,false,false,false,false,false, true,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 91.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 92.0,100.0,
		 52.0, 53.0,100.0,100.0,100.0, 100.0,100.0,100.0, 93.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		 50.0, 51.0,100.0,100.0,100.0, 100.0,100.0,100.0, 95.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 96.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 97.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 98.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 99.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellEightySeven){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* true_sinks_section_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false, true,false,false,false,
		false,false, true,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 91.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 92.0,100.0,
		 53.0, 53.0,100.0,100.0,100.0, 100.0,100.0,100.0, 93.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		 50.0, 51.0,100.0,100.0,100.0, 100.0,100.0,100.0, 95.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 96.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 97.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 98.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 99.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellEightyEight){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* true_sinks_section_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false, true,false,false,false,
		false,false, true,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 91.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 92.0,100.0,
		 53.0, 53.0,100.0,100.0,100.0, 100.0,100.0,100.0, 93.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		 50.0, 51.0,100.0,100.0,100.0, 100.0,100.0,100.0, 95.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 96.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 97.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 98.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 99.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,2,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,53.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellEightyNine){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* landsea_mask_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		 true,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 91.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 92.0,100.0,
		 53.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 93.0,100.0,
		100.0, 53.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0,100.0,100.0, 95.0,100.0,
		 50.0, 51.0,100.0,100.0,100.0, 100.0,100.0,100.0, 96.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 97.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 98.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 99.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0);
	alg1.setup_fields(orography_section,landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,53.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellNinety){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* landsea_mask_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 91.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 92.0,100.0,
		 53.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 93.0,100.0,
		100.0, 53.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0,100.0,100.0, 95.0,100.0,
		 50.0, 51.0,100.0,100.0,100.0, 100.0,100.0,100.0, 96.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 97.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 98.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 99.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0);
	alg1.setup_fields(orography_section,landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellNinetyOne){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* true_sinks_section_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		 true,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 91.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 92.0,100.0,
		 52.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 93.0,100.0,
		100.0, 53.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0,100.0,100.0, 95.0,100.0,
		 50.0, 51.0,100.0,100.0,100.0, 100.0,100.0,100.0, 96.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 97.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 98.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 99.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,2.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,53.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellNinetyTwo){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* true_sinks_section_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 91.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 92.0,100.0,
		 52.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 93.0,100.0,
		100.0, 53.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0,100.0,100.0, 95.0,100.0,
		 50.0, 51.0,100.0,100.0,100.0, 100.0,100.0,100.0, 96.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 97.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 98.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 99.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,2.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellNinetyThree){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* landsea_mask_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false, true,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 91.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 92.0,100.0,
		 53.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 93.0,100.0,
		100.0, 53.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0,100.0,100.0, 95.0,100.0,
		 50.0, 51.0,100.0,100.0,100.0, 100.0,100.0,100.0, 96.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 97.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 98.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 99.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0);
	alg1.setup_fields(orography_section,landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,92.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellNinetyFour){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	bool* landsea_mask_ten_by_ten = new bool[10*10] {
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		 true,false,false,false,false,false,false,false,false, true,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 91.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 92.0,100.0,
		 53.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 93.0,100.0,
		100.0, 53.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0,100.0,100.0, 95.0,100.0,
		 50.0, 51.0,100.0,100.0,100.0, 100.0,100.0,100.0, 96.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 97.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 98.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 99.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0);
	alg1.setup_fields(orography_section,landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,53.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellNinetyFive){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 91.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 92.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 93.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 95.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 96.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0,100.0,100.0, 97.0,100.0,
		100.0, 53.0,100.0, 51.0,100.0, 100.0,100.0,100.0, 98.0,100.0,
		100.0, 54.0,100.0, 50.0,100.0, 100.0,100.0,100.0, 99.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellNinetySix){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 91.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 92.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 93.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 95.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 96.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0,100.0,100.0, 97.0,100.0,
		100.0, 53.0,100.0, 51.0,100.0, 100.0,100.0,100.0, 98.0,100.0,
		 54.0,100.0,100.0, 50.0,100.0, 100.0,100.0,100.0, 99.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,54.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellNinetySeven){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 91.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 92.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 93.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 95.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 96.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0,100.0,100.0, 97.0,100.0,
		100.0, 53.0,100.0, 51.0,100.0, 100.0,100.0,100.0, 98.0,100.0,
		 54.0,100.0,100.0, 50.0,100.0, 100.0,100.0,100.0, 99.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0,true);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellNinetyEight){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 90.0,100.0,
		 54.0, 53.0,100.0,100.0,100.0, 100.0,100.0,100.0, 91.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0,100.0,100.0, 92.0,100.0,
		 50.0, 51.0,100.0,100.0,100.0, 100.0,100.0,100.0, 93.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 95.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 96.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 97.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 98.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 99.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellNinetyNine){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		 54.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0, 53.0,100.0,100.0,100.0, 100.0,100.0,100.0, 91.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0,100.0,100.0, 92.0,100.0,
		 50.0, 51.0,100.0,100.0,100.0, 100.0,100.0,100.0, 93.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 95.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 96.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 97.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 98.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 99.0,100.0};
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,54.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellOneHundred){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		 54.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0, 53.0,100.0,100.0,100.0, 100.0,100.0,100.0, 91.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0,100.0,100.0, 92.0,100.0,
		 50.0, 51.0,100.0,100.0,100.0, 100.0,100.0,100.0, 93.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 95.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 96.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 97.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 98.0,100.0,
		100.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 99.0,100.0};
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0,true);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellOneHundredAndOne){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0, 90.0,100.0,100.0,100.0, 100.0, 50.0,100.0, 54.0,100.0,
		100.0, 91.0,100.0,100.0,100.0, 100.0, 51.0,100.0, 53.0,100.0,
		100.0, 92.0,100.0,100.0,100.0, 100.0,100.0, 52.0,100.0,100.0,
		100.0, 93.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 94.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 95.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 96.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 97.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 98.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 99.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellOneHundredAndTwo){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0, 90.0,100.0,100.0,100.0, 100.0, 50.0,100.0,100.0, 54.0,
		100.0, 91.0,100.0,100.0,100.0, 100.0, 51.0,100.0, 53.0,100.0,
		100.0, 92.0,100.0,100.0,100.0, 100.0,100.0, 52.0,100.0,100.0,
		100.0, 93.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 94.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 95.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 96.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 97.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 98.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 99.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0};
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,54.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellOneHundredAndThree){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0, 90.0,100.0,100.0,100.0, 100.0, 50.0,100.0,100.0, 54.0,
		100.0, 91.0,100.0,100.0,100.0, 100.0, 51.0,100.0, 53.0,100.0,
		100.0, 92.0,100.0,100.0,100.0, 100.0,100.0, 52.0,100.0,100.0,
		100.0, 93.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 94.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 95.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 96.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 97.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 98.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 99.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0};
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0,true);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellOneHundredAndFour){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0, 90.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 91.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 92.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 93.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 94.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 95.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 96.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 97.0,100.0,100.0,100.0, 100.0,100.0, 52.0,100.0,100.0,
		100.0, 98.0,100.0,100.0,100.0, 100.0, 51.0,100.0, 53.0,100.0,
		100.0, 99.0,100.0,100.0,100.0, 100.0, 50.0,100.0, 54.0,100.0 };
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellOneHundredAndFive){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0, 90.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 91.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 92.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 93.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 94.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 95.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 96.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 97.0,100.0,100.0,100.0, 100.0,100.0, 52.0,100.0,100.0,
		100.0, 98.0,100.0,100.0,100.0, 100.0, 51.0,100.0, 53.0,100.0,
		100.0, 99.0,100.0,100.0,100.0, 100.0, 50.0,100.0,100.0, 54.0};
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,54.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellOneHundredAndSix){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0, 90.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 91.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 92.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 93.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 94.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 95.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 96.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 97.0,100.0,100.0,100.0, 100.0,100.0, 52.0,100.0,100.0,
		100.0, 98.0,100.0,100.0,100.0, 100.0, 51.0,100.0, 53.0,100.0,
		100.0, 99.0,100.0,100.0,100.0, 100.0, 50.0,100.0,100.0, 54.0};
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,3,1.0,true);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellOneHundredAndSeven){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0, 90.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 91.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 92.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 93.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 94.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 95.0,100.0,100.0,100.0, 100.0,100.0, 54.0,100.0, 60.0,
		100.0, 96.0,100.0,100.0,100.0, 100.0, 51.0,100.0, 53.0,100.0,
		100.0, 97.0,100.0,100.0,100.0, 100.0, 52.0,100.0, 53.0,100.0,
		100.0, 98.0,100.0,100.0,100.0, 100.0, 51.0,100.0, 53.0,100.0,
		100.0, 99.0,100.0,100.0,100.0, 100.0, 50.0,100.0, 51.0,100.0};
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,5,8.5,true);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,99.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellOneHundredAndEight){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0, 90.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 91.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 92.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 93.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 94.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 95.0,100.0,100.0,100.0, 100.0,100.0, 54.0,100.0, 60.0,
		100.0, 96.0,100.0,100.0,100.0, 100.0, 51.0,100.0, 53.0,100.0,
		100.0, 97.0,100.0,100.0,100.0, 100.0, 52.0,100.0, 53.0,100.0,
		100.0, 98.0,100.0,100.0,100.0, 100.0, 51.0,100.0, 53.0,100.0,
		100.0, 99.0,100.0,100.0,100.0, 100.0, 50.0,100.0, 51.0,100.0};
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,5,8.0,true);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,60.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingSingleOrographyCellOneHundredAndNine){
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	double answer;
	debug = false;
	double* orography_section = new double[10*10] {
		100.0, 90.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 91.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 92.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 93.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 94.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 95.0,100.0,100.0,100.0, 100.0,100.0, 54.0,100.0, 60.0,
		100.0, 96.0,100.0,100.0,100.0, 100.0, 51.0,100.0, 53.0,100.0,
		100.0, 97.0,100.0,100.0,100.0, 100.0, 52.0,100.0, 53.0,100.0,
		100.0, 98.0,100.0,100.0,100.0, 100.0, 51.0,100.0, 53.0,100.0,
		100.0, 99.0,100.0,100.0,100.0, 100.0, 50.0,100.0, 51.0,100.0};
	auto alg1 = sink_filling_algorithm_1_latlon();
	alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,
					 add_slope_in,epsilon_in,4,8.0,true);
	alg1.setup_fields(orography_section,all_land_landsea_mask_ten_by_ten,
					  empty_true_sinks_section_ten_by_ten,
					  new latlon_grid_params(10,10,true));
	alg1.fill_sinks();
	answer = alg1.tarasov_get_area_height();
	ASSERT_EQ(answer,54.0);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingTwobyTwoCellGridPartitionTest){
	int nlat_course = 2;
	int nlon_course = 2;
	int nlat_fine = nlat_course*10;
	int nlon_fine = nlon_course*10;
	int scale_factor_lat = nlat_fine/nlat_course;
	int scale_factor_lon = nlon_fine/nlon_course;
	double* orography_out = new double [nlat_course*nlon_course];
	for (auto i = 0; i < nlat_course*nlon_course; i++) {
		orography_out[i] = -999.0;
	}
	double* orography_in = new double[nlat_fine*nlon_fine] {
		100.0,101.0,102.0,103.0,104.0,105.0,106.0,107.0,108.0,109.0, 200.0,201.0,202.0,203.0,204.0,205.0,206.0,207.0,208.0,209.0,
		110.0,111.0,112.0,113.0,114.0,115.0,116.0,117.0,118.0,119.0, 210.0,211.0,212.0,213.0,214.0,215.0,216.0,217.0,218.0,219.0,
		120.0,121.0,122.0,123.0,124.0,125.0,126.0,127.0,128.0,129.0, 220.0,221.0,222.0,223.0,224.0,225.0,226.0,227.0,228.0,229.0,
		130.0,131.0,132.0,133.0,134.0,135.0,136.0,137.0,138.0,139.0, 230.0,231.0,232.0,233.0,234.0,235.0,236.0,237.0,238.0,239.0,
		140.0,141.0,142.0,143.0,144.0,145.0,146.0,147.0,148.0,149.0, 240.0,241.0,242.0,243.0,244.0,245.0,246.0,247.0,248.0,249.0,
		150.0,151.0,152.0,153.0,154.0,155.0,156.0,157.0,158.0,159.0, 250.0,251.0,252.0,253.0,254.0,255.0,256.0,257.0,258.0,259.0,
		160.0,161.0,162.0,163.0,164.0,165.0,166.0,167.0,168.0,169.0, 260.0,261.0,262.0,263.0,264.0,265.0,266.0,267.0,268.0,269.0,
		170.0,171.0,172.0,173.0,174.0,175.0,176.0,177.0,178.0,179.0, 270.0,271.0,272.0,273.0,274.0,275.0,276.0,277.0,278.0,279.0,
		180.0,181.0,182.0,183.0,184.0,185.0,186.0,187.0,188.0,189.0, 280.0,281.0,282.0,283.0,284.0,285.0,286.0,287.0,288.0,289.0,
		190.0,191.0,192.0,193.0,194.0,195.0,196.0,197.0,198.0,199.0, 290.0,291.0,292.0,293.0,294.0,295.0,296.0,297.0,298.0,299.0,

		300.0,301.0,302.0,303.0,304.0,305.0,306.0,307.0,308.0,309.0, 400.0,401.0,402.0,403.0,404.0,405.0,406.0,407.0,408.0,409.0,
		310.0,311.0,312.0,313.0,314.0,315.0,316.0,317.0,318.0,319.0, 410.0,411.0,412.0,413.0,414.0,415.0,416.0,417.0,418.0,419.0,
		320.0,321.0,322.0,323.0,324.0,325.0,326.0,327.0,328.0,329.0, 420.0,421.0,422.0,423.0,424.0,425.0,426.0,427.0,428.0,429.0,
		330.0,331.0,332.0,333.0,334.0,335.0,336.0,337.0,338.0,339.0, 430.0,431.0,432.0,433.0,434.0,435.0,436.0,437.0,438.0,439.0,
		340.0,341.0,342.0,343.0,344.0,345.0,346.0,347.0,348.0,349.0, 440.0,441.0,442.0,443.0,444.0,445.0,446.0,447.0,448.0,449.0,
		350.0,351.0,352.0,353.0,354.0,355.0,356.0,357.0,358.0,359.0, 450.0,451.0,452.0,453.0,454.0,455.0,456.0,457.0,458.0,459.0,
		360.0,361.0,362.0,363.0,364.0,365.0,366.0,367.0,368.0,369.0, 460.0,461.0,462.0,463.0,464.0,465.0,466.0,467.0,468.0,469.0,
		370.0,371.0,372.0,373.0,374.0,375.0,376.0,377.0,378.0,379.0, 470.0,471.0,472.0,473.0,474.0,475.0,476.0,477.0,478.0,479.0,
		380.0,381.0,382.0,383.0,384.0,385.0,386.0,387.0,388.0,389.0, 480.0,481.0,482.0,483.0,484.0,485.0,486.0,487.0,488.0,489.0,
		390.0,391.0,392.0,393.0,394.0,395.0,396.0,397.0,398.0,399.0, 490.0,491.0,492.0,493.0,494.0,495.0,496.0,497.0,498.0,499.0,

		};
	bool* landsea_in = new bool[nlat_fine*nlon_fine];
	bool* true_sinks_in = new bool[nlat_fine*nlon_fine];
	for (auto i = 0; i < nlat_fine*nlon_fine; i++) {
		landsea_in[i] = false;
		true_sinks_in[i] = false;
	}
	int iter_num = 1;
	bool test_failed = false;
	function<double(double*,bool*,bool*)> test_part = [&](double* orography_section,
														  bool* landsea_section,
														  bool* true_sinks_section) {
		for (auto i = 0; i < scale_factor_lat; i++){
			for (auto j = 0; j < scale_factor_lon; j++){
				if (orography_section[i*scale_factor_lon+j] !=
					100*iter_num + 10*i + j) test_failed = true;
			}
		}
		iter_num++;
		return 1.0;
	};
	partition_fine_orography(orography_in,landsea_in,true_sinks_in,
			 	 	 	 	 nlat_fine,nlon_fine,orography_out,
							 nlat_course,nlon_course,scale_factor_lat,
							 scale_factor_lon,test_part);
	ASSERT_FALSE(test_failed);
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingTwobyTwoCellGridTest){
	int method = 1;
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	int tarasov_separation_threshold_for_returning_to_same_edge_in = 5;
	double tarasov_min_path_length_in = 3.0;
	double  tarasov_include_corners_in_same_edge_criteria_in = true;
	int nlat_course = 2;
	int nlon_course = 2;
	int nlat_fine = nlat_course*10;
	int nlon_fine = nlon_course*10;
	double* orography_out = new double [nlat_course*nlon_course];
	for (auto i = 0; i < nlat_course*nlon_course; i++) {
		orography_out[i] = -999.0;
	}
	double* orography_in = new double[nlat_fine*nlon_fine] {
		100.0, 90.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0, 90.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 91.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0, 91.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 92.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0, 92.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 93.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0, 93.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 94.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0, 94.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 95.0,100.0,100.0,100.0, 100.0,100.0, 54.0,100.0, 60.0, 100.0, 95.0,100.0,100.0,100.0, 100.0,100.0, 54.0,100.0, 60.0,
		100.0, 96.0,100.0,100.0,100.0, 100.0, 51.0,100.0, 53.0,100.0, 100.0, 96.0,100.0,100.0,100.0, 100.0, 51.0,100.0, 53.0,100.0,
		100.0, 97.0,100.0,100.0,100.0, 100.0, 52.0,100.0, 53.0,100.0, 100.0, 97.0,100.0,100.0,100.0, 100.0, 52.0,100.0, 53.0,100.0,
		100.0, 98.0,100.0,100.0,100.0, 100.0, 51.0,100.0, 53.0,100.0, 100.0, 98.0,100.0,100.0,100.0, 100.0, 51.0,100.0, 53.0,100.0,
		100.0, 99.0,100.0,100.0,100.0, 100.0, 50.0,100.0, 51.0,100.0, 100.0, 99.0,100.0,100.0,100.0, 100.0, 50.0,100.0, 51.0,100.0,

		100.0, 90.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0, 90.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 91.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0, 91.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 92.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0, 92.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 93.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0, 93.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 94.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0, 94.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 95.0,100.0,100.0,100.0, 100.0,100.0, 54.0,100.0, 60.0, 100.0, 95.0,100.0,100.0,100.0, 100.0,100.0, 54.0,100.0, 60.0,
		100.0, 96.0,100.0,100.0,100.0, 100.0, 51.0,100.0, 53.0,100.0, 100.0, 96.0,100.0,100.0,100.0, 100.0, 51.0,100.0, 53.0,100.0,
		100.0, 97.0,100.0,100.0,100.0, 100.0, 52.0,100.0, 53.0,100.0, 100.0, 97.0,100.0,100.0,100.0, 100.0, 52.0,100.0, 53.0,100.0,
		100.0, 98.0,100.0,100.0,100.0, 100.0, 51.0,100.0, 53.0,100.0, 100.0, 98.0,100.0,100.0,100.0, 100.0, 51.0,100.0, 53.0,100.0,
		100.0, 99.0,100.0,100.0,100.0, 100.0, 50.0,100.0, 51.0,100.0, 100.0, 99.0,100.0,100.0,100.0, 100.0, 50.0,100.0, 51.0,100.0

		};
	bool* landsea_in = new bool[nlat_fine*nlon_fine];
	bool* true_sinks_in = new bool[nlat_fine*nlon_fine];
	for (auto i = 0; i < nlat_fine*nlon_fine; i++) {
		landsea_in[i] = false;
		true_sinks_in[i] = false;
	}
	latlon_upscale_orography(orography_in,nlat_fine,nlon_fine,orography_out,
							 nlat_course,nlon_course,method,landsea_in,true_sinks_in,
							 add_slope_in,epsilon_in,
							 tarasov_separation_threshold_for_returning_to_same_edge_in,
							 tarasov_min_path_length_in,
							 tarasov_include_corners_in_same_edge_criteria_in);
	for (auto i = 0; i < 4; i++){
		ASSERT_EQ(orography_out[i],60.0);
	}
}

TEST_F(FillSinksOrographyUpscalingTest,TestUpscalingFourbyFourCellGridTest){
	int method = 1;
	bool add_slope_in = false;
	double epsilon_in = 0.0;
	int tarasov_separation_threshold_for_returning_to_same_edge_in = 5;
	double tarasov_min_path_length_in = 3.0;
	double  tarasov_include_corners_in_same_edge_criteria_in = false;
	int nlat_course = 4;
	int nlon_course = 4;
	int nlat_fine = nlat_course*10;
	int nlon_fine = nlon_course*10;
	double* orography_out = new double [nlat_course*nlon_course];
	for (auto i = 0; i < nlat_course*nlon_course; i++) {
		orography_out[i] = -999.0;
	}
	double* orography_in = new double[nlat_fine*nlon_fine] {
		100.0, 90.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 90.0,100.0,100.0,  51.0,100.0, 65.0,100.0,100.0,
		100.0, 91.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 91.0,100.0,100.0,  52.0,100.0, 64.0,100.0,100.0,
		100.0, 92.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 92.0,100.0,100.0,  53.0,100.0, 63.0,100.0,100.0,
		100.0, 93.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 93.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 93.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 93.0,100.0,100.0,  54.0,100.0, 62.0,100.0,100.0,
		100.0, 94.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 94.0,100.0,100.0,  55.0,100.0, 61.0,100.0,100.0,
		100.0, 95.0,100.0,100.0,100.0, 100.0,100.0, 54.0,100.0, 60.0,  100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 95.0,100.0,100.0, 100.0, 58.0,100.0,100.0,100.0,
		100.0, 96.0,100.0,100.0,100.0, 100.0, 51.0,100.0, 53.0,100.0,  100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0, 78.0,  100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0, 78.0, 100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 97.0,100.0,100.0,100.0, 100.0, 52.0,100.0, 53.0,100.0,  100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0, 77.0,100.0,  100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0, 77.0,100.0, 100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 98.0,100.0,100.0,100.0, 100.0, 51.0,100.0, 53.0,100.0,  100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0, 76.0,100.0,  100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0, 76.0,100.0, 100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0, 99.0,100.0,100.0,100.0, 100.0, 50.0,100.0, 51.0,100.0,  100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0, 75.0,  100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0, 75.0,100.0, 100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,

		100.0,100.0, 90.0,100.0,100.0,  51.0,100.0, 65.0,100.0,100.0,  100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 90.0,100.0,100.0, 100.0, 55.0,100.0,100.0,100.0,  100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 91.0,100.0,100.0,  52.0,100.0, 64.0,100.0,100.0,  100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 91.0,100.0,100.0, 100.0, 54.0,100.0,100.0,100.0,  100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 92.0,100.0,100.0,  53.0,100.0, 63.0,100.0,100.0,  100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 92.0,100.0,100.0, 100.0, 53.0,100.0,100.0,100.0,  100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 93.0,100.0,100.0,  54.0,100.0, 62.0,100.0,100.0,  100.0,100.0, 93.0,100.0,100.0,  48.0, 49.0, 50.0, 51.0, 52.0, 100.0,100.0, 93.0,100.0,100.0, 100.0,100.0, 52.0, 51.0, 50.0,  100.0,100.0, 93.0,100.0,100.0, 100.0, 49.0, 50.0, 51.0, 52.0,
		100.0,100.0, 94.0,100.0,100.0, 100.0, 58.0,100.0,100.0,100.0,  100.0,100.0, 94.0,100.0, 47.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 94.0,100.0,100.0,  48.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,

		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 77.0,100.0,100.0, 100.0,100.0, 92.0,100.0,100.0, 100.0,100.0, 81.0,100.0,100.0, 100.0, 70.0,100.0,100.0,100.0,  100.0,100.0, 77.0,100.0,100.0, 100.0,100.0, 92.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 78.0, 79.0, 79.0,  79.0, 79.0, 80.0,100.0,100.0, 100.0,100.0, 87.0,100.0,100.0, 100.0, 88.0,100.0,100.0,100.0,  100.0,100.0, 78.0, 79.0, 79.0,  79.0, 79.0, 80.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 49.0,100.0,100.0,  100.0,100.0,100.0,100.0,100.0, 100.0, 88.0, 82.0,100.0,100.0, 100.0,100.0, 83.0,100.0,100.0, 100.0, 89.0,100.0,100.0,100.0,  100.0,100.0,100.0,100.0,100.0, 100.0, 88.0, 82.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 55.0,100.0,100.0,  100.0,100.0,100.0,100.0,100.0, 100.0, 91.0,100.0,100.0,100.0, 100.0,100.0, 86.0,100.0,100.0, 100.0, 84.0,100.0,100.0,100.0,  100.0,100.0,100.0,100.0,100.0, 100.0, 91.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 60.0,100.0,100.0,   78.0, 78.0,100.0,100.0,100.0, 100.0, 82.0,100.0,100.0,100.0, 100.0,100.0, 85.0,100.0,100.0, 100.0, 83.0,100.0,100.0,100.0,   78.0, 78.0,100.0,100.0,100.0, 100.0, 82.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 65.0,100.0,100.0,  100.0, 91.5,100.0,100.0,100.0, 100.0, 82.0,100.0,100.0,100.0, 100.0,100.0, 85.0,100.0,100.0, 100.0, 82.0,100.0,100.0,100.0,  100.0, 78.0,100.0,100.0,100.0, 100.0, 82.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 55.0,100.0,100.0,  100.0, 78.0,100.0,100.0,100.0, 100.0, 81.0,100.0,100.0,100.0, 100.0,100.0, 85.0,100.0,100.0, 100.0, 81.0,100.0,100.0,100.0,  100.0, 78.0,100.0,100.0,100.0, 100.0, 81.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 50.0,100.0,100.0,  100.0, 78.0,100.0, 79.0, 79.0,  80.0,100.0,100.0,100.0,100.0, 100.0,100.0, 85.0,100.0,100.0, 100.0, 82.0,100.0,100.0,100.0,  100.0, 78.0,100.0, 79.0, 79.0,  80.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 78.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0, 84.0, 85.0,100.0,100.0,  80.0,100.0,100.0,100.0,100.0,  100.0,100.0, 78.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
		100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 93.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  83.0,100.0, 90.0,100.0, 79.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,

		100.0,100.0,100.0,100.0, 59.0, 100.0,100.0,100.0, 90.0,100.0,  100.0,100.0,100.0,100.0, 53.0, 100.0,100.0,100.0, 90.0,100.0, 100.0,100.0,100.0,100.0, 53.0, 100.0,100.0,100.0, 90.0,100.0,  100.0,100.0,100.0,100.0, 53.0, 100.0,100.0,100.0, 90.0,100.0,
		100.0,100.0,100.0, 53.0,100.0, 100.0, 92.0, 91.0,100.0,100.0,  100.0,100.0,100.0, 53.0,100.0, 100.0, 92.0, 91.0,100.0,100.0, 100.0,100.0,100.0, 63.0,100.0, 100.0, 92.0, 91.0,100.0,100.0,  100.0,100.0,100.0, 63.0,100.0, 100.0, 92.0, 91.0,100.0,100.0,
		100.0,100.0, 52.0,100.0,100.0, 100.0, 93.0,100.0,100.0,100.0,  100.0,100.0, 52.0,100.0,100.0, 100.0, 93.0,100.0,100.0,100.0, 100.0,100.0, 52.0,100.0,100.0, 100.0, 93.0,100.0,100.0,100.0,  100.0,100.0, 52.0,100.0,100.0, 100.0, 93.0,100.0,100.0,100.0,
		100.0, 51.0,100.0,100.0,100.0, 100.0,100.0, 93.0, 94.0,100.0,  100.0, 51.0,100.0,100.0,100.0, 100.0,100.0, 93.0, 94.0,100.0, 100.0, 51.0,100.0,100.0,100.0, 100.0,100.0, 93.0, 94.0,100.0,  100.0, 51.0,100.0,100.0,100.0, 100.0,100.0, 93.0, 94.0,100.0,
		 50.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,   50.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,  50.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,   50.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
		100.0, 51.0, 52.0, 53.0, 54.0, 100.0,100.0, 95.0,100.0,100.0,  100.0, 51.0, 52.0, 53.0, 54.0, 100.0,100.0, 95.0,100.0,100.0, 100.0, 51.0, 52.0, 53.0, 54.0, 100.0,100.0, 95.0,100.0,100.0,  100.0, 51.0, 52.0, 53.0, 54.0, 100.0,100.0, 95.0,100.0,100.0,
		100.0, 52.0,100.0,100.0, 55.0, 100.0,100.0, 97.0, 96.0,100.0,  100.0, 52.0,100.0,100.0, 55.0, 100.0,100.0, 97.0, 96.0,100.0, 100.0, 52.0,100.0,100.0, 55.0, 100.0,100.0, 97.0, 96.0,100.0,  100.0, 52.0,100.0,100.0, 55.0, 100.0,100.0, 97.0, 96.0,100.0,
		100.0, 53.0,100.0,100.0, 56.0, 100.0, 98.0,100.0,100.0,100.0,  100.0, 53.0,100.0,100.0, 56.0, 100.0, 98.0,100.0,100.0,100.0, 100.0, 53.0,100.0,100.0, 56.0, 100.0, 98.0,100.0,100.0,100.0,  100.0, 53.0,100.0,100.0, 56.0, 100.0, 98.0,100.0,100.0,100.0,
		100.0, 54.0,100.0,100.0, 57.0, 100.0, 98.0,100.0,100.0,100.0,  100.0, 54.0,100.0,100.0, 57.0, 100.0, 98.0,100.0,100.0,100.0, 100.0, 54.0,100.0,100.0, 57.0, 100.0, 98.0,100.0,100.0,100.0,  100.0, 54.0,100.0,100.0, 57.0, 100.0, 98.0,100.0,100.0,100.0,
		100.0, 58.0,100.0,100.0, 60.0, 100.0,100.0, 99.0,100.0,100.0,  100.0, 58.0,100.0,100.0, 60.0, 100.0,100.0, 99.0,100.0,100.0, 100.0, 61.0,100.0,100.0, 60.0, 100.0,100.0, 99.0,100.0,100.0,  100.0, 60.0,100.0,100.0, 60.0, 100.0,100.0, 99.0,100.0,100.0
	};

	bool* landsea_in = new bool[nlat_fine*nlon_fine] {
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,  false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,  false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,  false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,  false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,  false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,  false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,  false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,  false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,  false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,  false,false,false,false,false,false,false,false,false,false,

		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,	  false,false,false,false,false,false,false,false,false,false,  false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,	  false,false,false,false,false,false,false,false,false,false,  false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,	  false,false,false,false,false,false,false,false,false,false,  false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,	  false,false,false,false,false,false,false,false,false,false,  false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,	  false,false,false,false,false,false,false,false,false,false,  false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,	  false,false,false,false,false,false,false,false,false,false,  false,false,false,false,false, true,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,	  false,false,false,false,false,false,false,false,false,false,  false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,	  false,false,false,false,false,false,false,false,false,false,  false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,	  false,false,false,false,false,false,false,false,false,false,  false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,	  false,false,false,false,false,false,false,false,false,false,  false,false,false,false,false,false,false,false,false,false,

		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false, true,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,

		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false
	};
	bool* true_sinks_in = new bool[nlat_fine*nlon_fine] {
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,

		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,	  false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,	  false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,	  false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,	  false,false,false,false,false,false,false, true,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false, true,false,false,false,false,false,	  false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,	  false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,	  false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,	  false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,	  false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,	  false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,

		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false, true,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,

		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false,   false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false
	};
	double* expected_output = new double [nlat_fine*nlon_fine] {
		60.0,78.0,78.0,65.0,
		99.0,52.0,55.0,52.0,
		80.0,91.5,87.0,91.0,
		58.0,53.0,60.0,60.0
	};
	latlon_upscale_orography(orography_in,nlat_fine,nlon_fine,orography_out,
							 nlat_course,nlon_course,method,landsea_in,true_sinks_in,
							 add_slope_in,epsilon_in,
							 tarasov_separation_threshold_for_returning_to_same_edge_in,
							 tarasov_min_path_length_in,
							 tarasov_include_corners_in_same_edge_criteria_in);
	for (auto i = 0; i < nlat_course*nlat_course; i++){
		EXPECT_EQ(orography_out[i],expected_output[i]);
	}
}


/*
 *  Class containing tests specific to algorithm 4
 */
class FillSinksAlgorithmFourTest : public ::testing::Test {

protected:
	//Declare assorted test data
	double* orography_in;
	double* orography_in_wrapped_sink;
	double* orography_in_prefer_non_diagonal_test;
	bool*   ls_data;
	bool*   ls_data_wrapped_sink;
	bool*   ls_data_prefer_non_diagonal_test;
	bool*   expected_completed_cells_in;
	bool*   expected_completed_cells_no_ls;
	double* expected_orography_queue_no_ls_mask;
	double* expected_orography_in;
	double* expected_rdirs_in;
	double* expected_rdirs_initial_no_ls_mask;
	double* expected_rdirs_no_ls_mask;
	double* expected_orography_no_ls_mask;
	int nlat = 10;
	int nlon = 10;
public:
	//Constructor
	FillSinksAlgorithmFourTest();
	//Destructor
	~FillSinksAlgorithmFourTest();

};

//Constructor; define assorted test data
FillSinksAlgorithmFourTest::FillSinksAlgorithmFourTest(){

	orography_in = new double[nlat*nlon] {
		0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
		0.0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,0.0,
		0.0,1.1,2.0,2.0,1.2,2.0,2.0,2.0,1.1,0.0,
	    0.0,1.1,2.0,3.0,3.0,1.3,3.0,2.0,1.1,0.0,
	    0.0,1.1,2.0,3.0,0.1,0.1,3.0,2.0,1.1,0.0,
	    0.0,1.1,2.0,3.0,0.1,0.1,3.0,2.0,1.1,0.0,
	    0.0,1.1,2.0,3.0,3.0,3.0,3.0,2.0,1.1,0.0,
	    0.0,1.1,2.0,2.0,2.0,2.0,2.0,2.0,1.1,0.0,
	    0.0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,0.0,
	    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

	orography_in_wrapped_sink = new double[nlat*nlon] {
		0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
		1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1,
		2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,1.2,
		1.3,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0,
		0.1,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,0.1,
		0.1,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,0.1,
		3.0,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0,
		2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,2.0,
		1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1,
	    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

	orography_in_prefer_non_diagonal_test = new double[nlat*nlon]{
		1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
		1.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,1.0,
		1.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,
		1.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,1.0,
		1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,
		1.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,1.0,
		1.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,
		1.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,1.0,
		1.0,0.0,0.0,1.0,1.0,0.0,1.0,0.0,0.0,1.0,
		1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};

	ls_data_prefer_non_diagonal_test = new bool[nlat*nlon]{
		false,false,false,false,false,false,false,false,false,false,
		false,true,true,true,false,false,true,true,true,false,
		false,true,true,true,true,false,false,true,true,false,
		false,true,true,true,true,true,false,false,true,false,
		false,true,true,true,true,true,true,false,false,false,
		false,true,true,true,true,true,false,false,true,false,
		false,true,true,true,true,false,false,true,true,false,
		false,true,true,true,false,false,true,true,true,false,
		false,true,true,false,false,true,false,true,true,false,
		false,false,false,false,false,false,false,false,false,false
	};

	ls_data = new bool[nlat*nlon]{
		false,false,false,false,false,false,false,false,true, false,
		false,false,false,false,false,false,false,false,true, false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,true, true, true, true, false,false,false,false,false,
		false,false,true, true, true, false,false,false,false,false,
		false,false,true, false,false,false,false,false,false,false,
		false,false,true, false,false,false,false,false,false,false,
		false,true, true, true, true, true, true, true,false,false,
		true, true, true, true, true, true, true, true,false,false};

	ls_data_wrapped_sink = new bool[nlat*nlon]{
		false,false,false,false,true,true,false,false,false,false,
		false,false,false,false,true,true,false,false,false,false,
		false,false,false,false,true,true,false,false,false,false,
		false,false,false,false,true,true,false,false,false,false,
		false,false,false,false,true,true,false,false,false,false,
		false,false,false,false,true,true,false,false,false,false,
		false,false,false,false,true,true,false,false,false,false,
		false,false,false,false,true,true,false,false,false,false,
		false,false,false,false,true,true,false,false,false,false,
		false,false,false,false,true,true,false,false,false,false};

	expected_completed_cells_in = new bool[nlat*nlon]{
		false,false,false,false,false,false,false,true,true,true,
		false,false,false,false,false,false,false,true,true,true,
		false,false,false,false,false,false,false,true,true,true,
		true,true,true,true,true,true,false,false,false,false,
		true,true,true,true,true,true,false,false,false,false,
		true,true,true,true,true,true,false,false,false,false,
		false,true,true,true,true,true,false,false,false,false,
		true,true,true,true,true,true,true,true,true,false,
		true,true,true,true,true,true,true,true,true,true,
		true,true,true,true,true,true,true,true,true,true};

	expected_completed_cells_no_ls = new bool[nlat*nlon]{
		true, true, true, true, true, true, true, true, true, true,
		true, false,false,false,false,false,false,false,false,true,
		true, false,false,false,false,false,false,false,false,true,
		true, false,false,false,false,false,false,false,false,true,
		true, false,false,false,false,false,false,false,false,true,
		true, false,false,false,false,false,false,false,false,true,
		true, false,false,false,false,false,false,false,false,true,
		true, false,false,false,false,false,false,false,false,true,
		true, false,false,false,false,false,false,false,false,true,
		true, true, true, true, true, true, true, true, true, true,
	};

	expected_orography_in = new double[nlat*nlon] {
	-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,0.0,-10.0,0.0,
	-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,1.1,-10.0,0.0,
	-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,2.0,1.1,0.0,
    0.0,1.1,2.0,3.0,3.0,1.3,-10.0,-10.0,-10.0,-10.0,
    0.0,-10.0,-10.0,-10.0,-10.0,0.1,-10.0,-10.0,-10.0,-10.0,
    0.0,1.1,-10.0,-10.0,-10.0,0.1,-10.0,-10.0,-10.0,-10.0,
    -10.0,1.1,-10.0,3.0,3.0,3.0,-10.0,-10.0,-10.0,-10.0,
    0.0,1.1,-10.0,2.0,2.0,2.0,2.0,2.0,1.1,-10.0,
    0.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,1.1,0.0,
    -10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,0.0,0.0};

	expected_rdirs_in = new double[nlat*nlon] {
		0.0,0.0,0.0,0.0,0.0,0.0,0.0,6.0,0.0,4.0,
		0.0,0.0,0.0,0.0,0.0,0.0,0.0,9.0,0.0,7.0,
		0.0,0.0,0.0,0.0,0.0,0.0,0.0,9.0,8.0,7.0,
	    3.0,2.0,1.0,3.0,2.0,1.0,0.0,0.0,0.0,0.0,
	    6.0,0.0,0.0,0.0,0.0,4.0,0.0,0.0,0.0,0.0,
	    9.0,8.0,0.0,0.0,0.0,7.0,0.0,0.0,0.0,0.0,
	    0.0,9.0,0.0,9.0,8.0,7.0,0.0,0.0,0.0,0.0,
	    3.0,2.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,
	    2.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,3.0,
	    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.0,6.0
	};

	expected_orography_queue_no_ls_mask = new double[nlat*nlon]{
		0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
		0.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,0.0,
		0.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,0.0,
		0.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,0.0,
		0.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,0.0,
		0.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,0.0,
		0.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,0.0,
		0.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,0.0,
		0.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,0.0,
		0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
	};

	expected_rdirs_initial_no_ls_mask = new double[nlat*nlon]{
		4.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,6.0,
		4.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,6.0,
		4.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,6.0,
		4.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,6.0,
		4.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,6.0,
		4.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,6.0,
		4.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,6.0,
		4.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,6.0,
		4.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,6.0,
		4.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,6.0
	};

	expected_orography_no_ls_mask = new double[nlat*nlon] {
		0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
		0.0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,0.0,
		0.0,1.1,2.0,2.0,1.2,2.0,2.0,2.0,1.1,0.0,
	    0.0,1.1,2.0,3.0,3.0,1.3,3.0,2.0,1.1,0.0,
	    0.0,1.1,2.0,3.0,0.1,0.1,3.0,2.0,1.1,0.0,
	    0.0,1.1,2.0,3.0,0.1,0.1,3.0,2.0,1.1,0.0,
	    0.0,1.1,2.0,3.0,3.0,3.0,3.0,2.0,1.1,0.0,
	    0.0,1.1,2.0,2.0,2.0,2.0,2.0,2.0,1.1,0.0,
	    0.0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,0.0,
	    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

	expected_rdirs_no_ls_mask = new double[nlat*nlon]{
		4.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,6.0,
		4.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,9.0,6.0,
		4.0,7.0,7.0,7.0,7.0,7.0,7.0,9.0,9.0,6.0,
		4.0,7.0,7.0,9.0,8.0,7.0,4.0,9.0,9.0,6.0,
		4.0,7.0,7.0,6.0,9.0,8.0,7.0,9.0,9.0,6.0,
		4.0,7.0,7.0,9.0,9.0,8.0,7.0,9.0,9.0,6.0,
		4.0,7.0,7.0,9.0,9.0,8.0,7.0,9.0,9.0,6.0,
		4.0,7.0,7.0,1.0,1.0,1.0,1.0,9.0,9.0,6.0,
		4.0,7.0,1.0,1.0,1.0,1.0,1.0,1.0,9.0,6.0,
		4.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,6.0
	};
}

//Destructor; clean-up
FillSinksAlgorithmFourTest::~FillSinksAlgorithmFourTest(){
	delete [] orography_in;
	delete [] orography_in_wrapped_sink;
	delete [] ls_data;
	delete [] ls_data_wrapped_sink;
}

//Check that the find_initial_cell_flow_directions function works correctly
TEST_F(FillSinksAlgorithmFourTest,TestFindingInitialCellFlowDirections){
	auto orography = new field<double>(orography_in,new latlon_grid_params(nlat,nlon));
	auto* landsea_mask = new field<bool>(ls_data,new latlon_grid_params(nlat,nlon));
	auto alg4 = sink_filling_algorithm_4_latlon();
	auto lat = 3;
	auto lon = 2;
	auto answer1 = alg4.test_find_initial_cell_flow_direction(new latlon_coords(lat,lon),
															  new latlon_grid_params(nlat,nlon),
															  orography,landsea_mask);
	ASSERT_EQ(answer1,1);
	lat = 1;
	lon = 9;
	auto answer2 = alg4.test_find_initial_cell_flow_direction(new latlon_coords(lat,lon),
															  new latlon_grid_params(nlat,nlon),
															  orography,landsea_mask);
	ASSERT_EQ(answer2,7);
	lat = 1;
	lon = 7;
	auto answer3 = alg4.test_find_initial_cell_flow_direction(new latlon_coords(lat,lon),
															  new latlon_grid_params(nlat,nlon),
															  orography,landsea_mask);
	ASSERT_EQ(answer3,9);
	lat = 4;
	lon = 5;
	auto answer4 = alg4.test_find_initial_cell_flow_direction(new latlon_coords(lat,lon),
															  new latlon_grid_params(nlat,nlon),
															  orography,landsea_mask);
	ASSERT_EQ(answer4,4);
	lat = 0;
	lon = 0;
	auto answer5 = alg4.test_find_initial_cell_flow_direction(new latlon_coords(lat,lon),
															  new latlon_grid_params(nlat,nlon),
															  orography,landsea_mask);
	ASSERT_EQ(answer5,5);
	lat = 1;
	lon = 1;
	auto answer6 = alg4.test_find_initial_cell_flow_direction(new latlon_coords(lat,lon),
															  new latlon_grid_params(nlat,nlon),
															  orography,landsea_mask);
	ASSERT_EQ(answer6,5);
	lat = 9;
	lon = 0;
	auto answer7 = alg4.test_find_initial_cell_flow_direction(new latlon_coords(lat,lon),
															  new latlon_grid_params(nlat,nlon),
															  orography,landsea_mask);
	ASSERT_EQ(answer7,5);
	lat = 9;
	lon = 9;
	auto answer8 = alg4.test_find_initial_cell_flow_direction(new latlon_coords(lat,lon),
															  new latlon_grid_params(nlat,nlon),
															  orography,landsea_mask);
	ASSERT_EQ(answer8,6);
	lat = 3;
	lon = 3;
	auto answer9 = alg4.test_find_initial_cell_flow_direction(new latlon_coords(lat,lon),
														      new latlon_grid_params(nlat,nlon),
															  orography,landsea_mask);
	ASSERT_EQ(answer9,3);
}

//Check that the calculate_direction_from_neighbor_to_cell function works correctly
TEST_F(FillSinksAlgorithmFourTest,TestFindingDirectionToCell){
	 auto alg4 = sink_filling_algorithm_4_latlon();
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(new latlon_coords(11,11),
			 new latlon_coords(10,10),new latlon_grid_params(100,100)),7);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(new latlon_coords(11,10),
			 new latlon_coords(10,10),new latlon_grid_params(100,100)),8);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(new latlon_coords(11,9),
			 new latlon_coords(10,10),new latlon_grid_params(100,100)),9);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(new latlon_coords(10,11),
			 new latlon_coords(10,10),new latlon_grid_params(100,100)),4);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(new latlon_coords(10,10),
			 new latlon_coords(10,10),new latlon_grid_params(100,100)),5);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(new latlon_coords(10,9),
			 new latlon_coords(10,10),new latlon_grid_params(100,100)),6);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(new latlon_coords(9,11),
			 new latlon_coords(10,10),new latlon_grid_params(100,100)),1);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(new latlon_coords(9,10),
			 new latlon_coords(10,10),new latlon_grid_params(100,100)),2);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(new latlon_coords(9,9),
			 new latlon_coords(10,10),new latlon_grid_params(100,100)),3);
	 ASSERT_THROW(alg4.test_calculate_direction_from_neighbor_to_cell(new latlon_coords(9,8),
			 new latlon_coords(10,10),new latlon_grid_params(100,100)),runtime_error);
	 ASSERT_THROW(alg4.test_calculate_direction_from_neighbor_to_cell(new latlon_coords(8,11),
			 new latlon_coords(10,10),new latlon_grid_params(100,100)),runtime_error);
	 ASSERT_THROW(alg4.test_calculate_direction_from_neighbor_to_cell(new latlon_coords(10,12),
			 new latlon_coords(10,10),new latlon_grid_params(100,100)),runtime_error);
	 ASSERT_THROW(alg4.test_calculate_direction_from_neighbor_to_cell(new latlon_coords(12,9),
			 new latlon_coords(10,10),new latlon_grid_params(100,100)),runtime_error);
	 //Test a few longitudinal wrapping scenarios
	 ASSERT_THROW(alg4.test_calculate_direction_from_neighbor_to_cell(new latlon_coords(12,0),
			 new latlon_coords(10,99),new latlon_grid_params(100,100)),runtime_error);
	 ASSERT_THROW(alg4.test_calculate_direction_from_neighbor_to_cell(new latlon_coords(8,99),
			 new latlon_coords(10,0),new latlon_grid_params(100,100)),runtime_error);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(new latlon_coords(9, 99),
			 new latlon_coords(10,0),new latlon_grid_params(100,100)),3);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(new latlon_coords(10,99),
			 new latlon_coords(10,0),new latlon_grid_params(100,100)),6);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(new latlon_coords(11,99),
			 new latlon_coords(10,0),new latlon_grid_params(100,100)),9);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(new latlon_coords(9,0),
			 new latlon_coords(10,99),new latlon_grid_params(100,100)),1);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(new latlon_coords(10,0),
			 new latlon_coords(10,99),new latlon_grid_params(100,100)),4);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(new latlon_coords(11,0),
			 new latlon_coords(10,99),new latlon_grid_params(100,100)),7);
}

//Test the find_initial_cell_flow_direction function again now with the prefer_non_diagonals flag set to true
TEST_F(FillSinksAlgorithmFourTest,TestFindingInitialCellFlowDirectionsPreferNonDiagonals){
	auto* orography = new field<double>(orography_in_prefer_non_diagonal_test,new latlon_grid_params(nlat,nlon));
	auto* landsea_mask = new field<bool>(ls_data_prefer_non_diagonal_test,new latlon_grid_params(nlat,nlon));
	auto alg4 = sink_filling_algorithm_4_latlon();
	auto lat = 0;
	auto lon = 0;
	auto answer1 = alg4.test_find_initial_cell_flow_direction(new latlon_coords(lat,lon),
			new latlon_grid_params(nlat,nlon),orography,landsea_mask,true);
	ASSERT_EQ(answer1,3);
	lat = 0;
	lon = 1;
	auto answer2 = alg4.test_find_initial_cell_flow_direction(new latlon_coords(lat,lon),
			new latlon_grid_params(nlat,nlon),orography,landsea_mask,true);
	ASSERT_EQ(answer2,2);
	lat = 0;
	lon = 2;
	auto answer3 = alg4.test_find_initial_cell_flow_direction(new latlon_coords(lat,lon),
			new latlon_grid_params(nlat,nlon),orography,landsea_mask,true);
	ASSERT_EQ(answer3,2);
	// check one case with prefer non diagonals switch to false to check expected difference occurs
	lat = 0;
	lon = 2;
	auto answer4 = alg4.test_find_initial_cell_flow_direction(new latlon_coords(lat,lon),
			new latlon_grid_params(nlat,nlon),orography,landsea_mask,false);
	ASSERT_EQ(answer4,1);
	lat = 0;
	lon = 9;
	auto answer5 = alg4.test_find_initial_cell_flow_direction(new latlon_coords(lat,lon),
			new latlon_grid_params(nlat,nlon),orography,landsea_mask,true);
	ASSERT_EQ(answer5,1);
	lat = 2;
	lon = 9;
	auto answer6 = alg4.test_find_initial_cell_flow_direction(new latlon_coords(lat,lon),
			new latlon_grid_params(nlat,nlon),orography,landsea_mask,true);
	ASSERT_EQ(answer6,4);
	lat = 4;
	lon = 7;
	auto answer7 = alg4.test_find_initial_cell_flow_direction(new latlon_coords(lat,lon),
			new latlon_grid_params(nlat,nlon),orography,landsea_mask,true);
	ASSERT_EQ(answer7,4);
	lat = 9;
	lon = 9;
	auto answer8 = alg4.test_find_initial_cell_flow_direction(new latlon_coords(lat,lon),
			new latlon_grid_params(nlat,nlon),orography,landsea_mask,true);
	ASSERT_EQ(answer8,7);
	lat = 7;
	lon = 5;
	auto answer9 = alg4.test_find_initial_cell_flow_direction(new latlon_coords(lat,lon),
			new latlon_grid_params(nlat,nlon),orography,landsea_mask,true);
	ASSERT_EQ(answer9,2);
	lat = 9;
	lon = 0;
	auto answer10 = alg4.test_find_initial_cell_flow_direction(new latlon_coords(lat,lon),
			new latlon_grid_params(nlat,nlon),orography,landsea_mask,true);
	ASSERT_EQ(answer10,9);
	lat = 9;
	lon = 2;
	auto answer11 = alg4.test_find_initial_cell_flow_direction(new latlon_coords(lat,lon),
			new latlon_grid_params(nlat,nlon),orography,landsea_mask,true);
	ASSERT_EQ(answer11,8);
	lat = 5;
	lon = 0;
	auto answer12 = alg4.test_find_initial_cell_flow_direction(new latlon_coords(lat,lon),
			new latlon_grid_params(nlat,nlon),orography,landsea_mask,true);
	ASSERT_EQ(answer12,6);
	//Test another with prefer non diagonals set to false
	lat = 5;
	lon = 0;
	auto answer13 = alg4.test_find_initial_cell_flow_direction(new latlon_coords(lat,lon),
			new latlon_grid_params(nlat,nlon),orography,landsea_mask,false);
	ASSERT_EQ(answer13,9);
	lat = 1;
	lon = 1;
	auto answer14 = alg4.test_find_initial_cell_flow_direction(new latlon_coords(lat,lon),
			new latlon_grid_params(nlat,nlon),orography,landsea_mask,true);
	ASSERT_EQ(answer14,5);
	lat = 2;
	lon = 2;
	auto answer15 = alg4.test_find_initial_cell_flow_direction(new latlon_coords(lat,lon),
			new latlon_grid_params(nlat,nlon),orography,landsea_mask,true);
	ASSERT_EQ(answer15,5);
}

//Test the add_edge_cells_to_q function for algorithm 4 without supplying
//a land sea mask; the set up is similar to the main code although the
//object tested is just the function add_edge_cell_to_q
TEST_F(FillSinksAlgorithmFourTest,TestAddingEdgesToQueueNoLSMask){
	auto* orography = new field<double>(orography_in,new latlon_grid_params(nlat,nlon));
	auto* completed_cells = new field<bool>(new latlon_grid_params(nlat,nlon));
	completed_cells->set_all(false);
	auto * catchment_nums = new field<int>(new latlon_grid_params(nlat,nlon));
	catchment_nums->set_all(0);
	bool* landsea_mask = nullptr;
	auto* next_cell_lat_index = new field<int>(new latlon_grid_params(nlat,nlon));
	auto* next_cell_lon_index = new field<int>(new latlon_grid_params(nlat,nlon));
	field<double>* rdirs = new field<double>(new latlon_grid_params(nlat,nlon));
	rdirs->set_all(0.0);
	auto alg4 = sink_filling_algorithm_4_latlon(orography,new latlon_grid_params(nlat,nlon),
												completed_cells,landsea_mask,false,
												catchment_nums,false,false,
												next_cell_lat_index,next_cell_lon_index,
												nullptr,rdirs);
	alg4.test_add_edge_cells_to_q();
	priority_cell_queue q = alg4.get_q();
	//Output the data in the queue to an array in order to validate it
	auto count = 0;
	field<double> orography_in_queue(new latlon_grid_params(nlat,nlon));
	orography_in_queue.set_all(-10.0);
	while(!q.empty()){
		auto coords = q.top()->get_cell_coords();
		orography_in_queue(coords) = q.top()->get_orography();
		q.pop();
		count++;
	}
	EXPECT_TRUE(orography_in_queue ==
				field<double>(expected_orography_queue_no_ls_mask,new latlon_grid_params(nlat,nlon)));
	EXPECT_TRUE(*rdirs == field<double>(expected_rdirs_initial_no_ls_mask,new latlon_grid_params(nlat,nlon)));
	EXPECT_TRUE(*completed_cells ==
		        field<bool>(expected_completed_cells_no_ls,new latlon_grid_params(nlat,nlon)));
}

//Test the add_edge_cells_to_q function for algorithm 4; the set up is similar to the
//main code although the object tested is just the function add_edge_cell_to_q
TEST_F(FillSinksAlgorithmFourTest,TestAddingEdgesToQueue){
	auto orography = new field<double>(orography_in,new latlon_grid_params(nlat,nlon));
	auto* completed_cells = new field<bool>(new latlon_grid_params(nlat,nlon));
	completed_cells->set_all(false);
	bool* landsea_mask = ls_data;
	field<double>* rdirs = new field<double>(new latlon_grid_params(nlat,nlon));
	rdirs->set_all(0.0);
	auto* catchment_nums = new field<int>(new latlon_grid_params(nlat,nlon));
	auto* next_cell_lat_index = new field<int>(new latlon_grid_params(nlat,nlon));
	auto* next_cell_lon_index = new field<int>(new latlon_grid_params(nlat,nlon));
	catchment_nums->set_all(0);
	auto alg4 = sink_filling_algorithm_4_latlon(orography,new latlon_grid_params(nlat,nlon),
												completed_cells,landsea_mask,false,catchment_nums,
												false,false,next_cell_lat_index,next_cell_lon_index,
												nullptr,rdirs);
	alg4.test_add_edge_cells_to_q();
	priority_cell_queue q = alg4.get_q();
	//Output the data in the queue to an array in order to validate it
	auto count = 0;
	field<double> orography_in_queue(new latlon_grid_params(nlat,nlon));
	orography_in_queue.set_all(-10.0);
	while(!q.empty()){
		auto coords = q.top()->get_cell_coords();
		orography_in_queue(coords) = q.top()->get_orography();
		q.pop();
		count++;
	}
	auto expected_completed_cells = field<bool>(expected_completed_cells_in,new latlon_grid_params(nlat,nlon));
	EXPECT_TRUE(*completed_cells == expected_completed_cells);
	auto expected_orography_in_queue = field<double>(expected_orography_in,new latlon_grid_params(nlat,nlon));
	EXPECT_TRUE(orography_in_queue == expected_orography_in_queue);
	auto expected_rdirs = field<double>(expected_rdirs_in,new latlon_grid_params(nlat,nlon));
	EXPECT_TRUE((*rdirs) == expected_rdirs);
	EXPECT_EQ(count,35);
}

//Tests the full program against some pseudo-data checking both the output
//river direction and that the output orography hasn't changed
TEST_F(FillSinksAlgorithmFourTest,TestFillSinks){
	auto method = 4;
	bool* landsea_in = nullptr;
	bool* true_sinks_in = nullptr;
	double* rdirs_in = new double[nlat*nlon];
	int* next_cell_lat_index_in = new int[nlat*nlon];
	int* next_cell_lon_index_in = new int[nlat*nlon];
	int* catchment_nums_in = new int[nlat*nlon];
	for (auto i = 0; i < nlat*nlon; i++){
		rdirs_in[i] = 0.0;
		catchment_nums_in[i] = 0;
	}
	latlon_fill_sinks(orography_in,nlat,nlon,method,landsea_in,false,true_sinks_in,
					  false,0.0,next_cell_lat_index_in,next_cell_lon_index_in,
					  rdirs_in,catchment_nums_in,false);
	EXPECT_TRUE(field<double>(orography_in,new latlon_grid_params(nlat,nlon)) ==
			    field<double>(expected_orography_no_ls_mask,new latlon_grid_params(nlat,nlon)));
	EXPECT_TRUE(field<double>(rdirs_in,new latlon_grid_params(nlat,nlon)) ==
			    field<double>(expected_rdirs_no_ls_mask,new latlon_grid_params(nlat,nlon)));
}

/*
 * Class containing tests specific to algorithm 1 (though the code tested my well
 * be shared by the other algorithms)
 */

class FillSinksTest : public ::testing::Test {

protected:
	//Declare pseudo data and expected results
	double* orography_in;
	double* expected_orography_out;
	double* orography_in_wrapped_sink;
	double* orography_in_wrapped_sink_with_slope;
	double* orography_in_with_slope;
	double* expected_orography_wrapped_sink;
	double* expected_orography_wrapped_sink_ls_filled;
	double* expected_orography_in_queue_out;
	double* expected_orography_in_queue_out_with_true_sinks;
	double* expected_orography_wrapped_sink_ls_filled_with_slope;
	double* expected_orography_out_with_slope;
	double* adding_to_q_expected_orography;
	double* adding_to_q_expected_orography_basic_version;
	double* adding_true_sinks_to_q_expected_orography_basic_version;
	bool* adding_to_q_expected_completed_cells_basic_version;
	bool* adding_to_q_expected_completed_cells;
	bool* adding_true_sinks_to_q_expected_completed_cells_basic_version;
	bool* adding_true_sinks_to_q_expected_completed_cells;
	bool* ls_data;
	bool* ls_data_wrapped_sink;
	bool* true_sinks_input;
	bool* true_sinks_input_ls_mask;
	int nlat = 10;
	int nlon = 10;
public:
	//Constructor
	FillSinksTest();
	//Destructor
	~FillSinksTest();

};

//Class constructor; define pseudo data and expected results
FillSinksTest::FillSinksTest(){

	orography_in = new double[nlat*nlon] {
		0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
		0.0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,0.0,
		0.0,1.1,2.0,2.0,1.2,2.0,2.0,2.0,1.1,0.0,
	    0.0,1.1,2.0,3.0,3.0,1.3,3.0,2.0,1.1,0.0,
	    0.0,1.1,2.0,3.0,0.1,0.1,3.0,2.0,1.1,0.0,
	    0.0,1.1,2.0,3.0,0.1,0.1,3.0,2.0,1.1,0.0,
	    0.0,1.1,2.0,3.0,3.0,3.0,3.0,2.0,1.1,0.0,
	    0.0,1.1,2.0,2.0,2.0,2.0,2.0,2.0,1.1,0.0,
	    0.0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,0.0,
	    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

	orography_in_wrapped_sink = new double[nlat*nlon] {
		0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
		1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1,
		2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,1.2,
		1.3,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0,
		0.1,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,0.1,
		0.1,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,0.1,
		3.0,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0,
		2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,2.0,
		1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1,
	    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

	orography_in_wrapped_sink_with_slope = new double[nlat*nlon] {
		0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
		1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1,
		2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,1.5,
		1.45,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0,
		0.1,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,0.1,
		0.1,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,0.1,
		3.0,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0,
		2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,2.0,
		1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1,
	    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

	orography_in_with_slope = new double[nlat*nlon] {
		0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
		0.0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,0.0,
		0.0,1.1,2.0,2.0,1.2,2.0,2.0,2.0,1.1,0.0,
	    0.0,1.1,2.0,0.1,0.1,0.1,0.1,2.0,1.1,0.0,
	    0.0,1.1,2.0,0.1,0.1,0.1,0.1,1.4,1.1,0.0,
	    0.0,1.1,2.0,0.1,0.1,0.1,0.1,2.0,1.1,0.0,
	    0.0,1.1,2.0,0.1,0.1,0.1,0.1,2.0,1.1,0.0,
	    0.0,1.1,2.0,2.0,2.0,1.3,2.0,2.0,1.1,0.0,
	    0.0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,0.0,
	    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

	ls_data = new bool[nlat*nlon]{
		false,false,false,false,false,false,false,false,true, false,
		false,false,false,false,false,false,false,false,true, false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,true, true, true, true, false,false,false,false,false,
		false,false,true, true, true, false,false,false,false,false,
		false,false,true, false,false,false,false,false,false,false,
		false,false,true, false,false,false,false,false,false,false,
		false,true, true, true, true, true, true, true,false,false,
		true, true, true, true, true, true, true, true,false,false};

	ls_data_wrapped_sink = new bool[nlat*nlon]{
		false,false,false,false,true,true,false,false,false,false,
		false,false,false,false,true,true,false,false,false,false,
		false,false,false,false,true,true,false,false,false,false,
		false,false,false,false,true,true,false,false,false,false,
		false,false,false,false,true,true,false,false,false,false,
		false,false,false,false,true,true,false,false,false,false,
		false,false,false,false,true,true,false,false,false,false,
		false,false,false,false,true,true,false,false,false,false,
		false,false,false,false,true,true,false,false,false,false,
		false,false,false,false,true,true,false,false,false,false};

	true_sinks_input = new bool[nlat*nlon]{
		false,false,false,false,true,true,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,true,
		false,false,false,true,false,false,true,true,false,false,
		false,false,false,false,false,false,false,false,false,true,
		false,false,false,false,false,false,false,false,false,false,
		false,false,true,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,false,false,false,false,true,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
	};

	//This is not a ls mask; it is true sinks input to be used with
	//a ls mask
	true_sinks_input_ls_mask = new bool[nlat*nlon]{
		false,false,true,false,false,false,false,false,false,false,
		false,false,false,false,false,false,false,false,true,false,
		false,false,false,false,false,false,true,false,false,true,
		false,false,true,false,false,false,false,false,false,false,
		false,false,false,false,false,true,false,false,true,false,
		false,false,false,false,false,false,false,false,true,false,
		false,true,false,false,false,false,false,false,false,false,
		false,false,false,false,false,true,false,false,false,false,
		false,false,false,false,false,false,false,false,false,false,
		false,false,true,true,false,true,false,false,false,false,
	};

	adding_to_q_expected_completed_cells_basic_version = new bool[nlat*nlon]{
		true, true, true, true, true, true, true, true, true, true,
		true, false,false,false,false,false,false,false,false,true,
		true, false,false,false,false,false,false,false,false,true,
		true, false,false,false,false,false,false,false,false,true,
		true, false,false,false,false,false,false,false,false,true,
		true, false,false,false,false,false,false,false,false,true,
		true, false,false,false,false,false,false,false,false,true,
		true, false,false,false,false,false,false,false,false,true,
		true, false,false,false,false,false,false,false,false,true,
		true, true, true, true, true, true, true, true, true, true,
	};

	adding_true_sinks_to_q_expected_completed_cells_basic_version =
			new bool[nlat*nlon]{
		true, true, true, true, true, true, true, true, true, true,
		true, false,false,false,false,false,false,false,false,true,
		true, false,false,false,false,false,false,false,false,true,
		true, false,false,false,false,false,false,false,false,true,
		true, false,false,false,false,false,false,false,false,true,
		true, false,false,false,false,false,false,false,false,true,
		true, false,false,false,false,false,false,false,false,true,
		true, false,false,false,false,false,false,false,false,true,
		true, false,false,false,false,false,false,false,false,true,
		true, true, true, true, true, true, true, true, true, true,
	};

	adding_to_q_expected_completed_cells = new bool[nlat*nlon]{
		false,false,false,false,false,false,false,true, true, true,
		false,false,false,false,false,false,false,true, true, true,
		false,false,false,false,false,false,false,true, true, true,
		true, true, true, true, true, true, false,false,false,false,
		true, true, true, true, true, true, false,false,false,false,
		true, true, true, true, true, true, false,false,false,false,
		false,true, true, true, true, true, false,false,false,false,
		true, true, true, true, true, true, true, true, true, false,
		true, true, true, true, true, true, true, true, true, true,
		true, true, true, true, true, true, true, true, true, true,
	};

	adding_true_sinks_to_q_expected_completed_cells = new bool[nlat*nlon]{
		false,false,false,false,false,false,false,true, true, true,
		false,false,false,false,false,false,false,true, true, true,
		false,false,false,false,false,false,false,true, true, true,
		true, true, true, true, true, true, false,false,false,false,
		true, true, true, true, true, true, false,false,false,false,
		true, true, true, true, true, true, false,false,false,false,
		false,true, true, true, true, true, false,false,false,false,
		true, true, true, true, true, true, true, true, true, false,
		true, true, true, true, true, true, true, true, true, true,
		true, true, true, true, true, true, true, true, true, true,
	};

	adding_to_q_expected_orography_basic_version = new double[nlat*nlon]{
		0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
		0.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,0.0,
		0.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,0.0,
		0.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,0.0,
		0.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,0.0,
		0.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,0.0,
		0.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,0.0,
		0.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,0.0,
		0.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,0.0,
		0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
	};

	adding_true_sinks_to_q_expected_orography_basic_version = new double[nlat*nlon]{
		0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
		0.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,0.0,
		0.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,0.0,
		0.0,-10.0,-10.0,3.0,-10.0,-10.0,3.0,2.0,-10.0,0.0,
		0.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,0.0,
		0.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,0.0,
		0.0,-10.0,2.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,0.0,
		0.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,0.0,
		0.0,-10.0,-10.0,-10.0,-10.0,-10.0,1.1,-10.0,-10.0,0.0,
		0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
	};

	adding_to_q_expected_orography = new double[nlat*nlon]{
		0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,no_data_value,0.0,
		0.0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,no_data_value,0.0,
		0.0,1.1,2.0,2.0,1.2,2.0,2.0,2.0,1.1,0.0,
		0.0,1.1,2.0,3.0,3.0,1.3,3.0,2.0,1.1,0.0,
		0.0,no_data_value,no_data_value,no_data_value,no_data_value,0.1,3.0,2.0,1.1,0.0,
		0.0,1.1,no_data_value,no_data_value,no_data_value,0.1,3.0,2.0,1.1,0.0,
		0.0,1.1,no_data_value,3.0,3.0,3.0,3.0,2.0,1.1,0.0,
		0.0,1.1,no_data_value,2.0,2.0,2.0,2.0,2.0,1.1,0.0,
		0.0,no_data_value,no_data_value,no_data_value,no_data_value,no_data_value,no_data_value,no_data_value,1.1,0.0,
		no_data_value,no_data_value,no_data_value,no_data_value,no_data_value,no_data_value,no_data_value,no_data_value,0.0,0.0};

	expected_orography_in_queue_out = new double[nlat*nlon]{
		-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,  0.0,-10.0,  0.0,
		-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,  1.1,-10.0,  0.0,
		-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,  2.0,  1.1,  0.0,
		  0.0,  1.1,  2.0,  3.0,  3.0,  1.3,-10.0,-10.0,-10.0,-10.0,
		  0.0,-10.0,-10.0,-10.0,-10.0,  0.1,-10.0,-10.0,-10.0,-10.0,
		  0.0,  1.1,-10.0,-10.0,-10.0,  0.1,-10.0,-10.0,-10.0,-10.0,
		-10.0,  1.1,-10.0,  3.0,  3.0,  3.0,-10.0,-10.0,-10.0,-10.0,
		  0.0,  1.1,-10.0,  2.0,  2.0,  2.0,  2.0,  2.0,  1.1,-10.0,
		  0.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,  1.1,  0.0,
		-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,  0.0,  0.0};

	expected_orography_in_queue_out_with_true_sinks = new double[nlat*nlon]{
		-10.0,-10.0,0.0,-10.0,-10.0,-10.0,-10.0,  0.0,-10.0,  0.0,
		-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,  1.1,-10.0,  0.0,
		-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,  2.0,  2.0,  1.1,  0.0,
		  0.0,  1.1,  2.0,  3.0,  3.0,  1.3,-10.0,-10.0,-10.0,-10.0,
		  0.0,-10.0,-10.0,-10.0,-10.0,  0.1,-10.0,-10.0,1.1,-10.0,
		  0.0,  1.1,-10.0,-10.0,-10.0,  0.1,-10.0,-10.0,1.1,-10.0,
		-10.0,  1.1,-10.0,  3.0,  3.0,  3.0,-10.0,-10.0,-10.0,-10.0,
		  0.0,  1.1,-10.0,  2.0,  2.0,  2.0,  2.0,  2.0,  1.1,-10.0,
		  0.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,  1.1,  0.0,
		-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,  0.0,  0.0};

	expected_orography_out = new double[nlat*nlon] {
		0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
		0.0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,0.0,
		0.0,1.1,2.0,2.0,1.2,2.0,2.0,2.0,1.1,0.0,
	    0.0,1.1,2.0,3.0,3.0,1.3,3.0,2.0,1.1,0.0,
	    0.0,1.1,2.0,3.0,1.3,1.3,3.0,2.0,1.1,0.0,
	    0.0,1.1,2.0,3.0,1.3,1.3,3.0,2.0,1.1,0.0,
	    0.0,1.1,2.0,3.0,3.0,3.0,3.0,2.0,1.1,0.0,
	    0.0,1.1,2.0,2.0,2.0,2.0,2.0,2.0,1.1,0.0,
	    0.0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,0.0,
	    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

	expected_orography_wrapped_sink  = new double[nlat*nlon]{
		0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
		1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1,
		2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,1.2,
		1.3,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0,
		1.3,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,1.3,
		1.3,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,1.3,
		3.0,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0,
		2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,2.0,
		1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1,
		0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

	expected_orography_wrapped_sink_ls_filled  = new double[nlat*nlon]{
		0.0,0.0,0.0,0.0,no_data_value,no_data_value,0.0,0.0,0.0,0.0,
		1.1,1.1,1.1,1.1,no_data_value,no_data_value,1.1,1.1,1.1,1.1,
		2.0,2.0,2.0,1.1,no_data_value,no_data_value,1.1,2.0,2.0,1.2,
		1.3,3.0,2.0,1.1,no_data_value,no_data_value,1.1,2.0,3.0,3.0,
		1.3,3.0,2.0,1.1,no_data_value,no_data_value,1.1,2.0,3.0,1.3,
		1.3,3.0,2.0,1.1,no_data_value,no_data_value,1.1,2.0,3.0,1.3,
		3.0,3.0,2.0,1.1,no_data_value,no_data_value,1.1,2.0,3.0,3.0,
		2.0,2.0,2.0,1.1,no_data_value,no_data_value,1.1,2.0,2.0,2.0,
		1.1,1.1,1.1,1.1,no_data_value,no_data_value,1.1,1.1,1.1,1.1,
		0.0,0.0,0.0,0.0,no_data_value,no_data_value,0.0,0.0,0.0,0.0};

	expected_orography_wrapped_sink_ls_filled_with_slope = new double[nlat*nlon]{
		0.3,0.2,0.1,0.0,no_data_value,no_data_value,0.0,0.1,0.2,0.3,
		1.1,1.1,1.1,1.1,no_data_value,no_data_value,1.1,1.1,1.1,1.1,
		2.0,2.0,2.0,1.1,no_data_value,no_data_value,1.1,2.0,2.0,1.5,
		1.6,3.0,2.0,1.1,no_data_value,no_data_value,1.1,2.0,3.0,3.0,
		1.7,3.0,2.0,1.1,no_data_value,no_data_value,1.1,2.0,3.0,1.7,
		1.8,3.0,2.0,1.1,no_data_value,no_data_value,1.1,2.0,3.0,1.8,
		3.0,3.0,2.0,1.1,no_data_value,no_data_value,1.1,2.0,3.0,3.0,
		2.0,2.0,2.0,1.1,no_data_value,no_data_value,1.1,2.0,2.0,2.0,
		1.1,1.1,1.1,1.1,no_data_value,no_data_value,1.1,1.1,1.1,1.1,
		0.3,0.2,0.1,0.0,no_data_value,no_data_value,0.0,0.1,0.2,0.3};

	expected_orography_out_with_slope = new double[nlat*nlon] {
			0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
			0.0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,0.0,
			0.0,1.1,2.0,2.0,1.2,2.0,2.0,2.0,1.1,0.0,
		    0.0,1.1,2.0,1.3,1.3,1.3,1.4,2.0,1.1,0.0,
		    0.0,1.1,2.0,1.4,1.4,1.4,1.4,1.4,1.1,0.0,
		    0.0,1.1,2.0,1.5,1.5,1.5,1.5,2.0,1.1,0.0,
		    0.0,1.1,2.0,1.5,1.4,1.4,1.4,2.0,1.1,0.0,
		    0.0,1.1,2.0,2.0,2.0,1.3,2.0,2.0,1.1,0.0,
		    0.0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,0.0,
		    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

}

//Class destructor; clean up
FillSinksTest::~FillSinksTest(){
	delete [] orography_in;
	delete [] orography_in_wrapped_sink;
	delete [] ls_data;
	delete [] ls_data_wrapped_sink;
	delete [] orography_in_wrapped_sink_with_slope;
	delete [] orography_in_with_slope;
	delete [] expected_orography_out_with_slope;
}

//Test the add_cells_to_q function without a land sea mask; the set up is similar to
//the main code although the object tested is just the function add_edge_cell_to_q
TEST_F(FillSinksTest,TestAddingEdgesToQueue){
	auto orography = new field<double>(orography_in,new latlon_grid_params(nlat,nlon));
	auto* completed_cells = new field<bool>(new latlon_grid_params(nlat,nlon));
	completed_cells->set_all(false);
	bool* landsea_mask = nullptr;
	auto alg1 = sink_filling_algorithm_1_latlon(orography,new latlon_grid_params(nlat,nlon),
										 	 	completed_cells,landsea_mask,true);
 	alg1.test_add_edge_cells_to_q();
	priority_cell_queue q = alg1.get_q();
 	EXPECT_TRUE(*completed_cells ==
 			    field<bool>(adding_to_q_expected_completed_cells_basic_version,
 			    		    new latlon_grid_params(nlat,nlon)));
	//Output the data in the queue to an array in order to validate it
 	auto count = 0;
 	field<double> orography_in_queue(new latlon_grid_params(nlat,nlon));
 	orography_in_queue.set_all(-10.0);
 	while(!q.empty()){
 		auto coords = q.top()->get_cell_coords();
 		orography_in_queue(coords) = q.top()->get_orography();
 		q.pop();
 		count++;
 	}
 	EXPECT_TRUE(orography_in_queue ==
 			    field<double>(adding_to_q_expected_orography_basic_version,new latlon_grid_params(nlat,nlon)));
 	EXPECT_EQ(count,36);
}

//Test the add_cells_to_q function with a land sea mask; the set up is similar to
//the main code although the object tested is just the function add_edge_cell_to_q
TEST_F(FillSinksTest,TestAddingEdgesToQueueWithLSMask){
	auto orography = new field<double>(orography_in,new latlon_grid_params(nlat,nlon));
	auto* completed_cells = new field<bool>(new latlon_grid_params(nlat,nlon));
	completed_cells->set_all(false);
	bool* landsea_mask = ls_data;
	auto alg1 = sink_filling_algorithm_1_latlon(orography,new latlon_grid_params(nlat,nlon),
												completed_cells,landsea_mask,true);
	alg1.test_add_edge_cells_to_q();
	priority_cell_queue q = alg1.get_q();
	EXPECT_TRUE(*completed_cells == field<bool>(adding_to_q_expected_completed_cells,new latlon_grid_params(nlat,nlon)));
	EXPECT_TRUE(*orography == field<double>(adding_to_q_expected_orography,new latlon_grid_params(nlat,nlon)));
	//Output the data in the queue to an array in order to validate it
	auto count = 0;
	field<double> orography_in_queue(new latlon_grid_params(nlat,nlon));
	orography_in_queue.set_all(-10.0);
	while(!q.empty()){
		auto coords = q.top()->get_cell_coords();
		orography_in_queue(coords) = q.top()->get_orography();
		q.pop();
		count++;
	}
	EXPECT_TRUE(orography_in_queue ==
			    field<double>(expected_orography_in_queue_out,new latlon_grid_params(nlat,nlon)));
	EXPECT_EQ(count,35);
}

//Test the add_true_sinks_to_q function without a land sea mask; the set up is similar to
//the main code although the object tested is just the function add_edge_cell_to_q
TEST_F(FillSinksTest,TestAddingTrueSinksToQueue){
	auto orography = new field<double>(orography_in,new latlon_grid_params(nlat,nlon));
	auto* completed_cells = new field<bool>(new latlon_grid_params(nlat,nlon));
	completed_cells->set_all(false);
	bool* landsea_mask = nullptr;
	auto alg1 = sink_filling_algorithm_1_latlon(orography,new latlon_grid_params(nlat,nlon),completed_cells,
										 	 	landsea_mask,true,false,0.0,true_sinks_input);
 	alg1.test_add_edge_cells_to_q();
 	alg1.test_add_true_sinks_to_q();
	priority_cell_queue q = alg1.get_q();
 	EXPECT_TRUE(*completed_cells ==
 			    field<bool>(adding_true_sinks_to_q_expected_completed_cells_basic_version,
 			    		new latlon_grid_params(nlat,nlon)));
	//Output the data in the queue to an array in order to validate it
 	auto count = 0;
 	field<double> orography_in_queue(new latlon_grid_params(nlat,nlon));
 	orography_in_queue.set_all(-10.0);
 	while(!q.empty()){
 		auto coords = q.top()->get_cell_coords();
 		orography_in_queue(coords) = q.top()->get_orography();
 		q.pop();
 		count++;
 	}
 	EXPECT_TRUE(orography_in_queue ==
 			    field<double>(adding_true_sinks_to_q_expected_orography_basic_version,
 			    			  new latlon_grid_params(nlat,nlon)));
 	EXPECT_EQ(count,41);
}

//Test true sinks to the queue with a land sea mask; the set up is similar to
//the main code although the object tested is just the function add_edge_cell_to_q
TEST_F(FillSinksTest,TestAddingTrueSinksToQueueWithLSMask){
	auto orography = new field<double>(orography_in,new latlon_grid_params(nlat,nlon));
	auto* completed_cells = new field<bool>(new latlon_grid_params(nlat,nlon));
	completed_cells->set_all(false);
	bool* landsea_mask = ls_data;
	auto alg1 = sink_filling_algorithm_1_latlon(orography,new latlon_grid_params(nlat,nlon),completed_cells,
												landsea_mask,true,false,0.0,true_sinks_input_ls_mask);
	alg1.test_add_edge_cells_to_q();
 	alg1.test_add_true_sinks_to_q();
	priority_cell_queue q = alg1.get_q();
	EXPECT_TRUE(*completed_cells == field<bool>(adding_true_sinks_to_q_expected_completed_cells,
												new latlon_grid_params(nlat,nlon)));
	EXPECT_TRUE(*orography == field<double>(adding_to_q_expected_orography,new latlon_grid_params(nlat,nlon)));
	//Output the data in the queue to an array in order to validate it
	auto count = 0;
	field<double> orography_in_queue(new latlon_grid_params(nlat,nlon));
	orography_in_queue.set_all(-10.0);
	while(!q.empty()){
		auto coords = q.top()->get_cell_coords();
		orography_in_queue(coords) = q.top()->get_orography();
		q.pop();
		count++;
	}
	EXPECT_TRUE(orography_in_queue ==
			    field<double>(expected_orography_in_queue_out_with_true_sinks,
			    			  new latlon_grid_params(nlat,nlon)));
	EXPECT_EQ(count,39);
}

//Test main fill_sinks code using algorithm 1 without a land-sea mask
TEST_F(FillSinksTest,TestFillSinks){
	auto method = 1;
	latlon_fill_sinks(orography_in,nlat,nlon,method);
	EXPECT_TRUE(field<double>(orography_in,new latlon_grid_params(nlat,nlon)) ==
			    field<double>(expected_orography_out,new latlon_grid_params(nlat,nlon)));
}

//Test main fill_sinks code using algorithm 1 with a land sea mask and
//longitudinal wrapping required
TEST_F(FillSinksTest,TestFillSinksWrappedSink){
	auto method = 1;
	latlon_fill_sinks(orography_in_wrapped_sink,nlat,nlon,method,ls_data_wrapped_sink);
	EXPECT_TRUE(field<double>(orography_in_wrapped_sink,new latlon_grid_params(nlat,nlon))
			    == field<double>(expected_orography_wrapped_sink_ls_filled,new latlon_grid_params(nlat,nlon)));
}

//Test main fill_sink code using algorithm with a land sea mask and
//longitudinal wrapping required and without setting sea points to
//no data
TEST_F(FillSinksTest,TestFillSinksWrappedSinkLSNotND){
	auto method = 1;
	latlon_fill_sinks(orography_in_wrapped_sink,nlat,nlon,method,ls_data_wrapped_sink,false);
	EXPECT_TRUE(field<double>(orography_in_wrapped_sink,new latlon_grid_params(nlat,nlon))
			    == field<double>(expected_orography_wrapped_sink,new latlon_grid_params(nlat,nlon)));
}

TEST_F(FillSinksTest,TestFillingSinkAddingSlope){
	auto method = 1;
	latlon_fill_sinks(orography_in_wrapped_sink_with_slope,nlat,nlon,method,ls_data_wrapped_sink,true,nullptr,true);
	EXPECT_TRUE(field<double>(orography_in_wrapped_sink_with_slope,new latlon_grid_params(nlat,nlon)).almost_equal(
				field<double>(expected_orography_wrapped_sink_ls_filled_with_slope,new latlon_grid_params(nlat,nlon))));
}

TEST_F(FillSinksTest,TestFillingSinkAddingSlopeMultipleEntryPoints){
	auto method = 1;
	latlon_fill_sinks(orography_in_with_slope,nlat,nlon,method,nullptr,true,nullptr,true);
	EXPECT_TRUE(field<double>(orography_in_with_slope,new latlon_grid_params(nlat,nlon)).almost_equal(
				field<double>(expected_orography_out_with_slope,new latlon_grid_params(nlat,nlon))));
}


/*
 * Tests of the field class
 */

class FieldTest : public ::testing::Test {

};

//Test overloaded equality operator
TEST_F(FieldTest,TestEquals){
	auto *test_array = new int[3*4] {1,2,3, 11,12,13, 101,102,103, 111,112,113};
	auto test_field = field<int>(test_array,new latlon_grid_params(4,3));
	auto *test_array2 = new int[3*4] {1,2,3, 11,12,13, 101,102,103, 111,112,113};
	auto test_field2 = field<int>(test_array2,new latlon_grid_params(4,3));
	auto *test_array3 = new int[3*4] {1,2,3, 11,12,13, 101,102,103, 111,112,114};
	auto test_field3 = field<int>(test_array3,new latlon_grid_params(4,3));
	auto *test_array4 = new int[3*4] {11,2,3, 11,120,13, 101,102,103, 111,112,114};
	auto test_field4 = field<int>(test_array4,new latlon_grid_params(4,3));
	EXPECT_TRUE(test_field == test_field);
	EXPECT_TRUE(test_field == test_field2);
	EXPECT_FALSE(test_field == test_field3);
	EXPECT_FALSE(test_field == test_field4);
	EXPECT_FALSE(test_field3 == test_field4);
}

//Test the set_all function
TEST_F(FieldTest,TestSetAll) {
	auto test_field = field<int>(new latlon_grid_params(9,9));
	test_field.set_all(16);
	EXPECT_EQ(test_field(new latlon_coords(8,8)),16);
	EXPECT_EQ(test_field(new latlon_coords(3,3)),16);
	EXPECT_EQ(test_field(new latlon_coords(3,8)),16);
	EXPECT_EQ(test_field(new latlon_coords(8,3)),16);
	EXPECT_EQ(test_field(new latlon_coords(0,0)),16);
}

//Test indexing on the left hand side of an expression
TEST_F(FieldTest,TestLHSIndexing){
	auto test_field = field<int>(new latlon_grid_params(9,9));
	test_field.set_all(16);
	test_field(new latlon_coords(4,3)) = -13;
	EXPECT_EQ(test_field.get_array()[9*4+3],-13);
	test_field(new latlon_coords(0,0)) = -35;
	EXPECT_EQ(test_field.get_array()[0],-35);
	test_field(new latlon_coords(0,0)) = 12;
	EXPECT_EQ(test_field.get_array()[0],12);
	test_field(new latlon_coords(8,8)) = 7;
	EXPECT_EQ(test_field.get_array()[9*8+8],7);
	EXPECT_EQ(test_field.get_array()[9*3+3],16);
}

//Test indexing on the right hand side of an expression
TEST_F(FieldTest,TestRHSIndexing){
	auto *test_array = new int[3*4] {1,2,3, 11,12,13, 101,102,103, 111,112,113};
	auto test_field = field<int>(test_array,new latlon_grid_params(4,3));
	EXPECT_EQ(test_field(new latlon_coords(2,2)),103);
	EXPECT_EQ(test_field(new latlon_coords(2,1)),102);
	EXPECT_EQ(test_field(new latlon_coords(1,2)),13);
	EXPECT_EQ(test_field(new latlon_coords(3,0)),111);
	EXPECT_EQ(test_field(new latlon_coords(2,0)),101);
	EXPECT_EQ(test_field(new latlon_coords(3,2)),113);
	EXPECT_EQ(test_field(new latlon_coords(0,2)),3);
	EXPECT_EQ(test_field(new latlon_coords(1,1)),12);
	EXPECT_EQ(test_field(new latlon_coords(0,1)),2);
	EXPECT_EQ(test_field(new latlon_coords(0,0)),1);
	delete [] test_array;
}

//test the get_neighbor_coords routine
TEST_F(FieldTest,TestGetneighbors){
	auto *test_array = new double[10*10];
	for (auto i = 0; i < 10*10; ++i){
		test_array[i] = i;
	}
	auto test_field = field<double>(test_array,new latlon_grid_params(10,10));
	auto neighbors_coords = test_field.get_neighbors_coords(new latlon_coords(2,7));
	auto expectation_0 = latlon_coords(1,6);
	auto expectation_1 = latlon_coords(1,7);
	auto expectation_2 = latlon_coords(1,8);
	auto expectation_3 = latlon_coords(2,6);
	auto expectation_4 = latlon_coords(2,8);
	auto expectation_5 = latlon_coords(3,6);
	auto expectation_6 = latlon_coords(3,7);
	auto expectation_7 = latlon_coords(3,8);
	EXPECT_EQ(expectation_0,*dynamic_cast<latlon_coords*>((*neighbors_coords)[0]));
	EXPECT_EQ(expectation_1,*dynamic_cast<latlon_coords*>((*neighbors_coords)[1]));
	EXPECT_EQ(expectation_2,*dynamic_cast<latlon_coords*>((*neighbors_coords)[2]));
	EXPECT_EQ(expectation_3,*dynamic_cast<latlon_coords*>((*neighbors_coords)[3]));
	EXPECT_EQ(expectation_4,*dynamic_cast<latlon_coords*>((*neighbors_coords)[4]));
	EXPECT_EQ(expectation_5,*dynamic_cast<latlon_coords*>((*neighbors_coords)[5]));
	EXPECT_EQ(expectation_6,*dynamic_cast<latlon_coords*>((*neighbors_coords)[6]));
	EXPECT_EQ(expectation_7,*dynamic_cast<latlon_coords*>((*neighbors_coords)[7]));
}

//Test the get_neighbors_coords routines for algorithm four (which puts the non-diagonal
//neighbors at the back of the vector so they processed first)
TEST_F(FieldTest,TestGetneighborsAlgorithmFour){
	auto *test_array = new double[10*10];
	for (auto i = 0; i < 10*10; ++i){
		test_array[i] = i;
	}
	auto test_field = field<double>(test_array,new latlon_grid_params(10,10));
	auto neighbors_coords = test_field.get_neighbors_coords(new latlon_coords(2,7),4);
	auto expectation_0 = latlon_coords(1,6);
	auto expectation_1 = latlon_coords(1,7);
	auto expectation_2 = latlon_coords(1,8);
	auto expectation_3 = latlon_coords(2,6);
	auto expectation_4 = latlon_coords(2,8);
	auto expectation_5 = latlon_coords(3,6);
	auto expectation_6 = latlon_coords(3,7);
	auto expectation_7 = latlon_coords(3,8);
	EXPECT_EQ(expectation_0,*dynamic_cast<latlon_coords*>((*neighbors_coords)[0]));
	EXPECT_EQ(expectation_2,*dynamic_cast<latlon_coords*>((*neighbors_coords)[1]));
	EXPECT_EQ(expectation_5,*dynamic_cast<latlon_coords*>((*neighbors_coords)[2]));
	EXPECT_EQ(expectation_7,*dynamic_cast<latlon_coords*>((*neighbors_coords)[3]));
	EXPECT_EQ(expectation_1,*dynamic_cast<latlon_coords*>((*neighbors_coords)[4]));
	EXPECT_EQ(expectation_6,*dynamic_cast<latlon_coords*>((*neighbors_coords)[5]));
	EXPECT_EQ(expectation_3,*dynamic_cast<latlon_coords*>((*neighbors_coords)[6]));
	EXPECT_EQ(expectation_4,*dynamic_cast<latlon_coords*>((*neighbors_coords)[7]));
}

} // close namespace

//The main function; runs all the tests
int main(int argc, char **argv){
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
