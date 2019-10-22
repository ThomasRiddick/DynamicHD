#ifndef FILTER_OUT_SHALLOW_LAKES_HPP_
#define FILTER_OUT_SHALLOW_LAKES_HPP_

void latlon_filter_out_shallow_lakes(double* unfilled_orography,double* filled_orography,
                                     double minimum_depth_threshold,int nlat_in,int nlon_in);

#endif /* FILTER_OUT_SHALLOW_LAKES_HPP_ */
