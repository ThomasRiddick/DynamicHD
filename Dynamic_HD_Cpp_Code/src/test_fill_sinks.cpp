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
#include "sink_filling_algorithm.hpp"
#include <iostream>
using namespace std;

namespace unittests {

//A global variable equal to the smallest possible double used as a non data value
const double no_data_value = numeric_limits<double>::lowest();

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
	auto orography = new field<double>(orography_in,nlat,nlon);
	auto* landsea_mask = new field<bool>(ls_data,nlat,nlon);
	auto alg4 = sink_filling_algorithm_4();
	auto lat = 3;
	auto lon = 2;
	auto answer1 = alg4.test_find_initial_cell_flow_direction(lat,lon,nlat,nlon,orography,landsea_mask);
	ASSERT_EQ(answer1,1);
	lat = 1;
	lon = 9;
	auto answer2 = alg4.test_find_initial_cell_flow_direction(lat,lon,nlat,nlon,orography,landsea_mask);
	ASSERT_EQ(answer2,7);
	lat = 1;
	lon = 7;
	auto answer3 = alg4.test_find_initial_cell_flow_direction(lat,lon,nlat,nlon,orography,landsea_mask);
	ASSERT_EQ(answer3,9);
	lat = 4;
	lon = 5;
	auto answer4 = alg4.test_find_initial_cell_flow_direction(lat,lon,nlat,nlon,orography,landsea_mask);
	ASSERT_EQ(answer4,4);
	lat = 0;
	lon = 0;
	auto answer5 = alg4.test_find_initial_cell_flow_direction(lat,lon,nlat,nlon,orography,landsea_mask);
	ASSERT_EQ(answer5,5);
	lat = 1;
	lon = 1;
	auto answer6 = alg4.test_find_initial_cell_flow_direction(lat,lon,nlat,nlon,orography,landsea_mask);
	ASSERT_EQ(answer6,5);
	lat = 9;
	lon = 0;
	auto answer7 = alg4.test_find_initial_cell_flow_direction(lat,lon,nlat,nlon,orography,landsea_mask);
	ASSERT_EQ(answer7,5);
	lat = 9;
	lon = 9;
	auto answer8 = alg4.test_find_initial_cell_flow_direction(lat,lon,nlat,nlon,orography,landsea_mask);
	ASSERT_EQ(answer8,6);
	lat = 3;
	lon = 3;
	auto answer9 = alg4.test_find_initial_cell_flow_direction(lat,lon,nlat,nlon,orography,landsea_mask);
	ASSERT_EQ(answer9,3);
}

//Check that the calculate_direction_from_neighbor_to_cell function works correctly
TEST_F(FillSinksAlgorithmFourTest,TestFindingDirectionToCell){
	 auto alg4 = sink_filling_algorithm_4();
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(11,11,10,10,100,100),7);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(11,10,10,10,100,100),8);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(11,9,10,10,100,100),9);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(10,11,10,10,100,100),4);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(10,10,10,10,100,100),5);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(10,9, 10,10,100,100),6);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(9,11, 10,10,100,100),1);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(9,10, 10,10,100,100),2);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(9,9, 10,10,100,100),3);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(9,8, 10,10,100,100),5);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(8,11, 10,10,100,100),5);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(10,12, 10,10,100,100),5);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(12,9, 10,10,100,100),5);
	 //Test a few longitudinal wrapping scenarios
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(12,0, 10,99,100,100),5);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(8,99, 10,0,100,100),5);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(9, 99, 10,0,100,100),3);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(10,99, 10,0,100,100),6);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(11,99, 10,0,100,100),9);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(9,0, 10,99,100,100),1);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(10,0, 10,99,100,100),4);
	 EXPECT_EQ(alg4.test_calculate_direction_from_neighbor_to_cell(11,0, 10,99,100,100),7);
}

//Test the find_initial_cell_flow_direction function again now with the prefer_non_diagonals flag set to true
TEST_F(FillSinksAlgorithmFourTest,TestFindingInitialCellFlowDirectionsPreferNonDiagonals){
	auto* orography = new field<double>(orography_in_prefer_non_diagonal_test,nlat,nlon);
	auto* landsea_mask = new field<bool>(ls_data_prefer_non_diagonal_test,nlat,nlon);
	auto alg4 = sink_filling_algorithm_4();
	auto lat = 0;
	auto lon = 0;
	auto answer1 = alg4.test_find_initial_cell_flow_direction(lat,lon,nlat,nlon,orography,landsea_mask,true);
	ASSERT_EQ(answer1,3);
	lat = 0;
	lon = 1;
	auto answer2 = alg4.test_find_initial_cell_flow_direction(lat,lon,nlat,nlon,orography,landsea_mask,true);
	ASSERT_EQ(answer2,2);
	lat = 0;
	lon = 2;
	auto answer3 = alg4.test_find_initial_cell_flow_direction(lat,lon,nlat,nlon,orography,landsea_mask,true);
	ASSERT_EQ(answer3,2);
	// check one case with prefer non diagonals switch to false to check expected difference occurs
	lat = 0;
	lon = 2;
	auto answer4 = alg4.test_find_initial_cell_flow_direction(lat,lon,nlat,nlon,orography,landsea_mask,false);
	ASSERT_EQ(answer4,1);
	lat = 0;
	lon = 9;
	auto answer5 = alg4.test_find_initial_cell_flow_direction(lat,lon,nlat,nlon,orography,landsea_mask,true);
	ASSERT_EQ(answer5,1);
	lat = 2;
	lon = 9;
	auto answer6 = alg4.test_find_initial_cell_flow_direction(lat,lon,nlat,nlon,orography,landsea_mask,true);
	ASSERT_EQ(answer6,4);
	lat = 4;
	lon = 7;
	auto answer7 = alg4.test_find_initial_cell_flow_direction(lat,lon,nlat,nlon,orography,landsea_mask,true);
	ASSERT_EQ(answer7,4);
	lat = 9;
	lon = 9;
	auto answer8 = alg4.test_find_initial_cell_flow_direction(lat,lon,nlat,nlon,orography,landsea_mask,true);
	ASSERT_EQ(answer8,7);
	lat = 7;
	lon = 5;
	auto answer9 = alg4.test_find_initial_cell_flow_direction(lat,lon,nlat,nlon,orography,landsea_mask,true);
	ASSERT_EQ(answer9,2);
	lat = 9;
	lon = 0;
	auto answer10 = alg4.test_find_initial_cell_flow_direction(lat,lon,nlat,nlon,orography,landsea_mask,true);
	ASSERT_EQ(answer10,9);
	lat = 9;
	lon = 2;
	auto answer11 = alg4.test_find_initial_cell_flow_direction(lat,lon,nlat,nlon,orography,landsea_mask,true);
	ASSERT_EQ(answer11,8);
	lat = 5;
	lon = 0;
	auto answer12 = alg4.test_find_initial_cell_flow_direction(lat,lon,nlat,nlon,orography,landsea_mask,true);
	ASSERT_EQ(answer12,6);
	//Test another with prefer non diagonals set to false
	lat = 5;
	lon = 0;
	auto answer13 = alg4.test_find_initial_cell_flow_direction(lat,lon,nlat,nlon,orography,landsea_mask,false);
	ASSERT_EQ(answer13,9);
	lat = 1;
	lon = 1;
	auto answer14 = alg4.test_find_initial_cell_flow_direction(lat,lon,nlat,nlon,orography,landsea_mask,true);
	ASSERT_EQ(answer14,5);
	lat = 2;
	lon = 2;
	auto answer15 = alg4.test_find_initial_cell_flow_direction(lat,lon,nlat,nlon,orography,landsea_mask,true);
	ASSERT_EQ(answer15,5);
}

//Test the add_edge_cells_to_q function for algorithm 4 without supplying
//a land sea mask; the set up is similar to the main code although the
//object tested is just the function add_edge_cell_to_q
TEST_F(FillSinksAlgorithmFourTest,TestAddingEdgesToQueueNoLSMask){
	auto orography = new field<double>(orography_in,nlat,nlon);
	auto* completed_cells = new field<bool>(nlat,nlon);
	completed_cells->set_all(false);
	auto * catchment_nums = new field<int>(nlat,nlon);
	catchment_nums->set_all(0);
	bool* landsea_mask = nullptr;
	field<double>* rdirs = new field<double>(nlat,nlon);
	rdirs->set_all(0.0);
	auto alg4 = sink_filling_algorithm_4(orography,nlat,nlon,rdirs,completed_cells,landsea_mask,
										 false,catchment_nums,false);
	alg4.test_add_edge_cells_to_q();
	priority_cell_queue q = alg4.get_q();
	//Output the data in the queue to an array in order to validate it
	auto count = 0;
	field<double> orography_in_queue(nlat,nlon);
	orography_in_queue.set_all(-10.0);
	while(!q.empty()){
		auto coords = q.top()->get_cell_coords();
		orography_in_queue(coords.first,coords.second) = q.top()->get_orography();
		q.pop();
		count++;
	}
	EXPECT_TRUE(orography_in_queue ==
				field<double>(expected_orography_queue_no_ls_mask,nlat,nlon));
	EXPECT_TRUE(*rdirs == field<double>(expected_rdirs_initial_no_ls_mask,nlat,nlon));
	EXPECT_TRUE(*completed_cells ==
		        field<bool>(expected_completed_cells_no_ls,nlat,nlon));
}

//Test the add_edge_cells_to_q function for algorithm 4; the set up is similar to the
//main code although the object tested is just the function add_edge_cell_to_q
TEST_F(FillSinksAlgorithmFourTest,TestAddingEdgesToQueue){
	auto orography = new field<double>(orography_in,nlat,nlon);
	auto* completed_cells = new field<bool>(nlat,nlon);
	completed_cells->set_all(false);
	bool* landsea_mask = ls_data;
	field<double>* rdirs = new field<double>(nlat,nlon);
	rdirs->set_all(0.0);
	auto* catchment_nums = new field<int>(nlat,nlon);
	catchment_nums->set_all(0);
	auto alg4 = sink_filling_algorithm_4(orography,nlat,nlon,rdirs,completed_cells,landsea_mask,
										 false,catchment_nums,false);
	alg4.test_add_edge_cells_to_q();
	priority_cell_queue q = alg4.get_q();
	//Output the data in the queue to an array in order to validate it
	auto count = 0;
	field<double> orography_in_queue(nlat,nlon);
	orography_in_queue.set_all(-10.0);
	while(!q.empty()){
		auto coords = q.top()->get_cell_coords();
		orography_in_queue(coords.first,coords.second) = q.top()->get_orography();
		q.pop();
		count++;
	}
	auto expected_completed_cells = field<bool>(expected_completed_cells_in,nlat,nlon);
	EXPECT_TRUE(*completed_cells == expected_completed_cells);
	auto expected_orography_in_queue = field<double>(expected_orography_in,nlat,nlon);
	EXPECT_TRUE(orography_in_queue == expected_orography_in_queue);
	auto expected_rdirs = field<double>(expected_rdirs_in,nlat,nlon);
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
	int* catchment_nums_in = new int[nlat*nlon];
	for (auto i = 0; i < nlat*nlon; i++){
		rdirs_in[i] = 0.0;
		catchment_nums_in[i] = 0;
	}
	fill_sinks(orography_in,nlat,nlon,method,landsea_in,false,true_sinks_in,rdirs_in,
			   catchment_nums_in);
	EXPECT_TRUE(field<double>(orography_in,nlat,nlon) ==
			    field<double>(expected_orography_no_ls_mask,nlat,nlon));
	EXPECT_TRUE(field<double>(rdirs_in,nlat,nlon) ==
			    field<double>(expected_rdirs_no_ls_mask,nlat,nlon));
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
	double* expected_orography_wrapped_sink;
	double* expected_orography_wrapped_sink_ls_filled;
	double* expected_orography_in_queue_out;
	double* expected_orography_in_queue_out_with_true_sinks;
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

}

//Class destructor; clean up
FillSinksTest::~FillSinksTest(){
	delete [] orography_in;
	delete [] orography_in_wrapped_sink;
	delete [] ls_data;
	delete [] ls_data_wrapped_sink;
}

//Test the add_cells_to_q function without a land sea mask; the set up is similar to
//the main code although the object tested is just the function add_edge_cell_to_q
TEST_F(FillSinksTest,TestAddingEdgesToQueue){
	auto orography = new field<double>(orography_in,nlat,nlon);
	auto* completed_cells = new field<bool>(nlat,nlon);
	completed_cells->set_all(false);
	bool* landsea_mask = nullptr;
	auto alg1 = sink_filling_algorithm_1(orography,nlat,nlon,completed_cells,landsea_mask,true);
 	alg1.test_add_edge_cells_to_q();
	priority_cell_queue q = alg1.get_q();
 	EXPECT_TRUE(*completed_cells ==
 			    field<bool>(adding_to_q_expected_completed_cells_basic_version,
 			    			nlat,nlon));
	//Output the data in the queue to an array in order to validate it
 	auto count = 0;
 	field<double> orography_in_queue(nlat,nlon);
 	orography_in_queue.set_all(-10.0);
 	while(!q.empty()){
 		auto coords = q.top()->get_cell_coords();
 		orography_in_queue(coords.first,coords.second) = q.top()->get_orography();
 		q.pop();
 		count++;
 	}
 	EXPECT_TRUE(orography_in_queue ==
 			    field<double>(adding_to_q_expected_orography_basic_version,nlat,nlon));
 	EXPECT_EQ(count,36);
}

//Test the add_cells_to_q function with a land sea mask; the set up is similar to
//the main code although the object tested is just the function add_edge_cell_to_q
TEST_F(FillSinksTest,TestAddingEdgesToQueueWithLSMask){
	auto orography = new field<double>(orography_in,nlat,nlon);
	auto* completed_cells = new field<bool>(nlat,nlon);
	completed_cells->set_all(false);
	bool* landsea_mask = ls_data;
	auto alg1 = sink_filling_algorithm_1(orography,nlat,nlon,completed_cells,landsea_mask,true);
	alg1.test_add_edge_cells_to_q();
	priority_cell_queue q = alg1.get_q();
	EXPECT_TRUE(*completed_cells == field<bool>(adding_to_q_expected_completed_cells,nlat,nlon));
	EXPECT_TRUE(*orography == field<double>(adding_to_q_expected_orography,nlat,nlon));
	//Output the data in the queue to an array in order to validate it
	auto count = 0;
	field<double> orography_in_queue(nlat,nlon);
	orography_in_queue.set_all(-10.0);
	while(!q.empty()){
		auto coords = q.top()->get_cell_coords();
		orography_in_queue(coords.first,coords.second) = q.top()->get_orography();
		q.pop();
		count++;
	}
	EXPECT_TRUE(orography_in_queue ==
			    field<double>(expected_orography_in_queue_out,nlat,nlon));
	EXPECT_EQ(count,35);
}

//Test the add_true_sinks_to_q function without a land sea mask; the set up is similar to
//the main code although the object tested is just the function add_edge_cell_to_q
TEST_F(FillSinksTest,TestAddingTrueSinksToQueue){
	auto orography = new field<double>(orography_in,nlat,nlon);
	auto* completed_cells = new field<bool>(nlat,nlon);
	completed_cells->set_all(false);
	bool* landsea_mask = nullptr;
	auto alg1 = sink_filling_algorithm_1(orography,nlat,nlon,completed_cells,landsea_mask,true,
			                             true_sinks_input);
 	alg1.test_add_edge_cells_to_q();
 	alg1.test_add_true_sinks_to_q();
	priority_cell_queue q = alg1.get_q();
 	EXPECT_TRUE(*completed_cells ==
 			    field<bool>(adding_true_sinks_to_q_expected_completed_cells_basic_version,
 			    			nlat,nlon));
	//Output the data in the queue to an array in order to validate it
 	auto count = 0;
 	field<double> orography_in_queue(nlat,nlon);
 	orography_in_queue.set_all(-10.0);
 	while(!q.empty()){
 		auto coords = q.top()->get_cell_coords();
 		orography_in_queue(coords.first,coords.second) = q.top()->get_orography();
 		q.pop();
 		count++;
 	}
 	EXPECT_TRUE(orography_in_queue ==
 			    field<double>(adding_true_sinks_to_q_expected_orography_basic_version,nlat,nlon));
 	EXPECT_EQ(count,41);
}

//Test true sinks to the queue with a land sea mask; the set up is similar to
//the main code although the object tested is just the function add_edge_cell_to_q
TEST_F(FillSinksTest,TestAddingTrueSinksToQueueWithLSMask){
	auto orography = new field<double>(orography_in,nlat,nlon);
	auto* completed_cells = new field<bool>(nlat,nlon);
	completed_cells->set_all(false);
	bool* landsea_mask = ls_data;
	auto alg1 = sink_filling_algorithm_1(orography,nlat,nlon,completed_cells,
										landsea_mask,true,true_sinks_input_ls_mask);
	alg1.test_add_edge_cells_to_q();
 	alg1.test_add_true_sinks_to_q();
	priority_cell_queue q = alg1.get_q();
	EXPECT_TRUE(*completed_cells == field<bool>(adding_true_sinks_to_q_expected_completed_cells,
												nlat,nlon));
	EXPECT_TRUE(*orography == field<double>(adding_to_q_expected_orography,nlat,nlon));
	//Output the data in the queue to an array in order to validate it
	auto count = 0;
	field<double> orography_in_queue(nlat,nlon);
	orography_in_queue.set_all(-10.0);
	while(!q.empty()){
		auto coords = q.top()->get_cell_coords();
		orography_in_queue(coords.first,coords.second) = q.top()->get_orography();
		q.pop();
		count++;
	}
	EXPECT_TRUE(orography_in_queue ==
			    field<double>(expected_orography_in_queue_out_with_true_sinks,
			    			  nlat,nlon));
	EXPECT_EQ(count,39);
}

//Test main fill_sinks code using algorithm 1 without a land-sea mask
TEST_F(FillSinksTest,TestFillSinks){
	auto method = 1;
	fill_sinks(orography_in,nlat,nlon,method);
	EXPECT_TRUE(field<double>(orography_in,nlat,nlon) ==
			    field<double>(expected_orography_out,nlat,nlon));
}

//Test main fill_sinks code using algorithm 1 with a land sea mask and
//longitudinal wrapping required
TEST_F(FillSinksTest,TestFillSinksWrappedSink){
	auto method = 1;
	fill_sinks(orography_in_wrapped_sink,nlat,nlon,method,ls_data_wrapped_sink);
	EXPECT_TRUE(field<double>(orography_in_wrapped_sink,nlat,nlon)
			    == field<double>(expected_orography_wrapped_sink_ls_filled,nlat,nlon));
}

//Test main fill_sink code using algorithm with a land sea mask and
//longitudinal wrapping required and without setting sea points to
//no data
TEST_F(FillSinksTest,TestFillSinksWrappedSinkLSNotND){
	auto method = 1;
	fill_sinks(orography_in_wrapped_sink,nlat,nlon,method,ls_data_wrapped_sink,false);
	EXPECT_TRUE(field<double>(orography_in_wrapped_sink,nlat,nlon)
			    == field<double>(expected_orography_wrapped_sink,nlat,nlon));
}

/*
 * Tests of the field class
 */

class FieldTest : public ::testing::Test {

};

//Test overloaded equality operator
TEST_F(FieldTest,TestEquals){
	auto *test_array = new int[3*4] {1,2,3, 11,12,13, 101,102,103, 111,112,113};
	auto test_field = field<int>(test_array,4,3);
	auto *test_array2 = new int[3*4] {1,2,3, 11,12,13, 101,102,103, 111,112,113};
	auto test_field2 = field<int>(test_array2,4,3);
	auto *test_array3 = new int[3*4] {1,2,3, 11,12,13, 101,102,103, 111,112,114};
	auto test_field3 = field<int>(test_array3,4,3);
	auto *test_array4 = new int[3*4] {11,2,3, 11,120,13, 101,102,103, 111,112,114};
	auto test_field4 = field<int>(test_array4,4,3);
	EXPECT_TRUE(test_field == test_field);
	EXPECT_TRUE(test_field == test_field2);
	EXPECT_FALSE(test_field == test_field3);
	EXPECT_FALSE(test_field == test_field4);
	EXPECT_FALSE(test_field3 == test_field4);
}

//Test the set_all function
TEST_F(FieldTest,TestSetAll) {
	auto test_field = field<int>(9,9);
	test_field.set_all(16);
	EXPECT_EQ(test_field(8,8),16);
	EXPECT_EQ(test_field(3,3),16);
	EXPECT_EQ(test_field(3,8),16);
	EXPECT_EQ(test_field(8,3),16);
	EXPECT_EQ(test_field(0,0),16);
}

//Test indexing on the left hand side of an expression
TEST_F(FieldTest,TestLHSIndexing){
	auto test_field = field<int>(9,9);
	test_field.set_all(16);
	test_field(4,3) = -13;
	EXPECT_EQ(test_field.get_array()[9*4+3],-13);
	test_field(0,0) = -35;
	EXPECT_EQ(test_field.get_array()[0],-35);
	test_field(0,0) = 12;
	EXPECT_EQ(test_field.get_array()[0],12);
	test_field(8,8) = 7;
	EXPECT_EQ(test_field.get_array()[9*8+8],7);
	EXPECT_EQ(test_field.get_array()[9*3+3],16);
}

//Test indexing on the right hand side of an expression
TEST_F(FieldTest,TestRHSIndexing){
	auto *test_array = new int[3*4] {1,2,3, 11,12,13, 101,102,103, 111,112,113};
	auto test_field = field<int>(test_array,4,3);
	EXPECT_EQ(test_field(2,2),103);
	EXPECT_EQ(test_field(2,1),102);
	EXPECT_EQ(test_field(1,2),13);
	EXPECT_EQ(test_field(3,0),111);
	EXPECT_EQ(test_field(2,0),101);
	EXPECT_EQ(test_field(3,2),113);
	EXPECT_EQ(test_field(0,2),3);
	EXPECT_EQ(test_field(1,1),12);
	EXPECT_EQ(test_field(0,1),2);
	EXPECT_EQ(test_field(0,0),1);
	delete [] test_array;
}

//test the get_neighbor_coords routine
TEST_F(FieldTest,TestGetneighbors){
	auto *test_array = new double[10*10];
	for (auto i = 0; i < 10*10; ++i){
		test_array[i] = i;
	}
	auto test_field = field<double>(test_array,10,10);
	auto neighbors_coords = test_field.get_neighbors_coords(pair<int,int>(2,7));
	auto expectation_0 = pair<int,int>(1,6);
	auto expectation_1 = pair<int,int>(1,7);
	auto expectation_2 = pair<int,int>(1,8);
	auto expectation_3 = pair<int,int>(2,6);
	auto expectation_4 = pair<int,int>(2,8);
	auto expectation_5 = pair<int,int>(3,6);
	auto expectation_6 = pair<int,int>(3,7);
	auto expectation_7 = pair<int,int>(3,8);
	EXPECT_EQ(expectation_0,(*(*neighbors_coords)[0]));
	EXPECT_EQ(expectation_1,(*(*neighbors_coords)[1]));
	EXPECT_EQ(expectation_2,(*(*neighbors_coords)[2]));
	EXPECT_EQ(expectation_3,(*(*neighbors_coords)[3]));
	EXPECT_EQ(expectation_4,(*(*neighbors_coords)[4]));
	EXPECT_EQ(expectation_5,(*(*neighbors_coords)[5]));
	EXPECT_EQ(expectation_6,(*(*neighbors_coords)[6]));
	EXPECT_EQ(expectation_7,(*(*neighbors_coords)[7]));
}

//Test the get_neighbors_coords routines for algorithm four (which puts the non-diagonal
//neighbors at the back of the vector so they processed first)
TEST_F(FieldTest,TestGetneighborsAlgorithmFour){
	auto *test_array = new double[10*10];
	for (auto i = 0; i < 10*10; ++i){
		test_array[i] = i;
	}
	auto test_field = field<double>(test_array,10,10);
	auto neighbors_coords = test_field.get_neighbors_coords(pair<int,int>(2,7),4);
	auto expectation_0 = pair<int,int>(1,6);
	auto expectation_1 = pair<int,int>(1,7);
	auto expectation_2 = pair<int,int>(1,8);
	auto expectation_3 = pair<int,int>(2,6);
	auto expectation_4 = pair<int,int>(2,8);
	auto expectation_5 = pair<int,int>(3,6);
	auto expectation_6 = pair<int,int>(3,7);
	auto expectation_7 = pair<int,int>(3,8);
	EXPECT_EQ(expectation_0,(*(*neighbors_coords)[0]));
	EXPECT_EQ(expectation_2,(*(*neighbors_coords)[1]));
	EXPECT_EQ(expectation_5,(*(*neighbors_coords)[2]));
	EXPECT_EQ(expectation_7,(*(*neighbors_coords)[3]));
	EXPECT_EQ(expectation_1,(*(*neighbors_coords)[4]));
	EXPECT_EQ(expectation_6,(*(*neighbors_coords)[5]));
	EXPECT_EQ(expectation_3,(*(*neighbors_coords)[6]));
	EXPECT_EQ(expectation_4,(*(*neighbors_coords)[7]));
}

} // close namespace

//The main function; runs all the tests
int main(int argc, char **argv){
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
