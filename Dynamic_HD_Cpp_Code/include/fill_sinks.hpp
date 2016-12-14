/*
 * fill_sinks.h
 *
 * Contains function declaration and global variables for the fill_sinks program
 *
 *  Created on: Mar 15, 2016
 *      Author: thomasriddick
 */

#ifndef FILL_SINKS_HPP_
#define FILL_SINKS_HPP_

//The main fill sinks routine; selects which algorithm to use and passes the input to it
void fill_sinks(double*, int, int, int,bool* = nullptr, bool = true,bool* = nullptr, double* = nullptr,
		int* = nullptr, bool=false);
//Interface function that takes integer as arguments and converts them to bool and also deal with creating a
//null pointer for the land sea mask argument of fill_sinks when the land sea mask is not being used
void fill_sinks_cython_interface(double*,int,int,int,int = 0,int* = nullptr,int = 1, int = 0, int* = nullptr,
		                         double* = nullptr, int* = nullptr, int=0);

#endif /* FILL_SINKS_HPP_ */
