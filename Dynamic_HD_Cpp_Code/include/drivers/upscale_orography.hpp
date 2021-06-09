/*
 * upscale_orography.hpp
 *
 *  Created on: Apr 4, 2017
 *      Author: thomasriddick
 */

#ifndef INCLUDE_UPSCALE_OROGRAPHY_HPP_
#define INCLUDE_UPSCALE_OROGRAPHY_HPP_

#include <functional>

using namespace std;

//A interface function for latlon_upscale_orography; converts necessary integer arguments
//to Boolean variables and calls latlon_upscale_orograph itself
void latlon_upscale_orography_cython_interface(double*, int, int, double*, int, int,
											   int, int*, int*, int, double, int,
											   double, int, int = 0);
//The main latitude-longitude Tarasov-style orography upscaling routine. Partitions the fine
//orography and loops over the cells created running each through the orography upscaling code
//for each
void latlon_upscale_orography(double*,int,int,double*,int,int,int,bool*,bool*,bool,double,int,
							 double,bool,bool=false);
//Split up a fine orography into chunks such that these chunks map to point in the grid of the
//course orography being generated and then run the supplied function over those points
void partition_fine_orography(double*,bool*,bool*,int,int,double*,int,int,int,int,
							  function<double(double*,bool*,bool*)>);

#endif /* INCLUDE_UPSCALE_OROGRAPHY_HPP_ */
