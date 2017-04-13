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

void latlon_upscale_orography_cython_interface(double*, int, int, double*, int, int,
											   int, int*, int*, int, double, int,
											   double, int, int = 0);
void latlon_upscale_orography(double*,int,int,double*,int,int,int,bool*,bool*,bool,double,int,
							 double,bool,bool=false);
void partition_fine_orography(double*,bool*,bool*,int,int,double*,int,int,int,int,
							  function<double(double*,bool*,bool*)>);

#endif /* INCLUDE_UPSCALE_OROGRAPHY_HPP_ */
