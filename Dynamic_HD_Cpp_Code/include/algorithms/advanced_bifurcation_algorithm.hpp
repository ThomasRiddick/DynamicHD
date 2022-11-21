#ifndef INCLUDE_ADVANCED_BIFURCATION_ALGORITHM_HPP_
#define INCLUDE_ADVANCED_BIFURCATION_ALGORITHM_HPP_

using namespace std;

class advanced_bifurcation_algorithm : virtual public bifurcation_algorithm {
  public:
    virtual ~advanced_bifurcation_algorithm() {};
  protected:
    void push_cell(coords* nbr_coords)
    field<double>* orography = nullptr;
};

class advanced_bifurcation_algorithm_latlon : public advanced_bifurcation_algorithm,
                                              public bifucation_algorithm_latlon {
  public:
    virtual ~basic_bifurcation_algorithm_latlon() {};
    void setup_fields(map<pair<int,int>,
                      vector<pair<int,int>>> river_mouths_in,
                      double* rdirs_in,
                      double* orography_in,
                      int* cumulative_flow_in,
                      int* number_of_outflows_in,
                      bool* landsea_mask_in,
                      grid_params* grid_params_in);
};


class advanced_bifurcation_algorithm_icon_single_index :
  public advanced_bifurcation_algorithm,
  public bifurcation_algorithm_icon_single_index {
  public:
    virtual ~basic_bifurcation_algorithm_icon_single_index() {};
    void setup_fields(map<int,vector<int>> river_mouths_in,
                      int* next_cell_index_in,
                      double* orography_in,
                      int* cumulative_flow_in,
                      int* number_of_outflows_in,
                      bool* landsea_mask_in,
                      grid_params* grid_params_in);
};

#endif /*INCLUDE_ADVANCED_BIFURCATION_ALGORITHM_HPP_*/
