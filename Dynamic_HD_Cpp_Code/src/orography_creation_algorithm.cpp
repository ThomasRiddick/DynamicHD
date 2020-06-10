#include "orography_creation_algorithm.hpp"
#include <iostream>
#include <string>
#include "grid.hpp"
using namespace std;

void orography_creation_algorithm::setup_flags(double sea_level_in){
  sea_level = sea_level_in;
}

void orography_creation_algorithm::setup_fields(bool* landsea_in,
                                                double* inclines_in,
                                                double* orography_in,
                                                grid_params* grid_params_in)
{
  _grid_params = grid_params_in;
  _grid = grid_factory(_grid_params);
  landsea = new field<bool>(landsea_in,_grid_params);
  inclines = new field<double>(inclines_in,_grid_params);
  orography = new field<double>(orography_in,_grid_params);
  completed_cells = new field<bool>(_grid_params);
  completed_cells->set_all(false);
}

orography_creation_algorithm::~orography_creation_algorithm()
{
  delete landsea;
  delete inclines;
  delete orography;
  delete completed_cells;
  delete _grid;
}

void orography_creation_algorithm::reset(){
  delete landsea;
  delete inclines;
  delete orography;
  delete completed_cells;
  delete _grid;
}

void orography_creation_algorithm::create_orography(){
  add_landsea_edge_cells_to_q();
  while (! q.empty()) {
    center_cell = q.top();
    q.pop();
    center_coords = center_cell->get_cell_coords();
    center_height = center_cell->get_orography();
    auto neighbors_coords = orography->get_neighbors_coords(center_coords,1);
    process_neighbors(neighbors_coords);
    delete neighbors_coords;
    delete center_cell;
  };
}

void orography_creation_algorithm::process_neighbors(vector<coords*>* neighbors_coords){
  //Loop through the neighbors on the supplied list
  while (! neighbors_coords->empty()) {
    coords* nbr_coords = neighbors_coords->back();
    if ((! (*completed_cells)(nbr_coords)) && (! (*landsea)(nbr_coords))) {
      double nbr_height = center_height + (*inclines)(nbr_coords);
      (*orography)(nbr_coords) = nbr_height;
      (*completed_cells)(nbr_coords) = true;
      q.push(new cell(nbr_height,nbr_coords->clone()));
    }
    neighbors_coords->pop_back();
    delete nbr_coords;
  }
}

void orography_creation_algorithm::add_landsea_edge_cells_to_q(){
  function<void(coords*)> add_edge_cell_to_q_func = [&](coords* coords_in){
    if((*landsea)(coords_in)){
      //set all points in the land sea mask as having been processed
      (*completed_cells)(coords_in) = true;
      (*orography)(coords_in) = sea_level;
      //Calculate and the process the neighbors of every landsea point and
      //add them to the queue (if they aren't already in the queue or themselves
      //land sea points
      auto neighbors_coords = orography->get_neighbors_coords(coords_in);
      while (!neighbors_coords->empty()){
        coords* nbr_coords = neighbors_coords->back();
        //If neither a land sea point nor a cell already in the queue
        //then add this cell to the queue
        if (!((*landsea)(nbr_coords) ||
          (*completed_cells)(nbr_coords))) {
          (*completed_cells)(nbr_coords) = true;
          double nbr_height = sea_level + (*inclines)(nbr_coords);
          (*orography)(nbr_coords) = nbr_height;
          q.push(new cell(nbr_height,nbr_coords->clone()));
        }
        neighbors_coords->pop_back();
        delete nbr_coords;
      }
      delete neighbors_coords;
    }
    delete coords_in;
  };
  _grid->for_all(add_edge_cell_to_q_func);
}
