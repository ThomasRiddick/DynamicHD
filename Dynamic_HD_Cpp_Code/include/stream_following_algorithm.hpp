/*
 * stream_follwing_algorithm.cpp
 *
 *  Created on: Feb 10, 2020
 *      Author: thomasriddick
 */

#include "coords.hpp"
#include "field.hpp"

#ifndef INCLUDE_STREAM_FLLOWING_ALGORITHM_HPP_
#define INCLUDE_STREAM_FLLOWING_ALGORITHM_HPP_

class stream_following_algorithm {
  public:
    virtual ~stream_following_algorithm() {delete downstream_cells;
                                           delete cells_with_loop;
                                           delete _grid;}
    void follow_streams_downstream();
    void follow_stream_downstream(coords* coords_in);
    void setup_fields(bool* cells_with_loop,
                      bool* downstream_cells,
                      grid_params* grid_params_in);
    virtual bool is_outflow(coords* coords_in) = 0;
    virtual coords* calculate_downstream_cell(coords* coords_in) = 0;
  protected:
    field<bool>* downstream_cells = nullptr;
    field<bool>* cells_with_loop = nullptr;
    grid* _grid = nullptr;
    grid_params* _grid_params = nullptr;
};

class dir_based_rdirs_stream_following_algorithm : public stream_following_algorithm {
  public:
    virtual ~dir_based_rdirs_stream_following_algorithm() {delete rdirs;}
    coords* calculate_downstream_cell(coords* coords_in);
    bool is_outflow(coords* coords_in);
    void setup_fields(double* rdirs_in,bool* cells_with_loop_in,
                      bool* downstream_cells_in,grid_params* grid_params_in);
  private:
    field<double>* rdirs = nullptr;
};

#endif /* INCLUDE_STREAM_FLLOWING_ALGORITHM_HPP_ */
