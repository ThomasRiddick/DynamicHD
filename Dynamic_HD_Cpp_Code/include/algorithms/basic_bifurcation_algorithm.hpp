#ifndef INCLUDE_BASIC_BIFURCATION_ALGORITHM_HPP_
#define INCLUDE_BASIC_BIFURCATION_ALGORITHM_HPP_

#include "algorithms/bifurcation_algorithm.hpp"

class basic_bifurcation_algorithm : virtual public bifurcation_algorithm {
  public:
    virtual ~basic_bifurcation_algorithm() {};
  protected:
    void push_cell(coords* nbr_coords);
    void push_coastal_cell(coords* cell_coords);
};

class basic_bifurcation_algorithm_latlon : public basic_bifurcation_algorithm,
                                           public bifurcation_algorithm_latlon {
  public:
    virtual ~basic_bifurcation_algorithm_latlon() {};
};

class basic_bifurcation_algorithm_icon_single_index :
  public basic_bifurcation_algorithm,
  public bifurcation_algorithm_icon_single_index {
  public:
    virtual ~basic_bifurcation_algorithm_icon_single_index() {};
};

#endif /*INCLUDE_BASIC_BIFURCATION_ALGORITHM_HPP_*/
