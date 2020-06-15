/*
 * depression_hierarchy_construction_algorithm.hpp
 *
 *  Created on: Feb 10, 2020
 *      Author: thomasriddick
 * Implements the algorithm of Barnes et al 2020
 * Computing water flow through complex landscapes – Part 2: Finding hierarchies in depressions and morphological segmentations
 * Earth Surface Dynamics
 */

typedef pair<int,int> outlet_label_type;

class outlet_label : public outlet_label_type {
  outlet_label(int depression_label_one,int depression_label_two){
    if (depression_label_one <= depression_label_two){
      outlet_label_type::pair(depression_label_one,depression_label_two);
    } else {
      outlet_label_type::pair(depression_label_two,depression_label_one);
    }
  }
}

class depression_hierarchy_construction_algorithm{
  protected:
    depression_hierarchy dh;
  public:
    void construct_hierarchy();
}

class depression_hierarchy : disjoint_set{
  protected:
    element* ocean = new element(0);
  public:
    element* get_element(int label_in){
      if(label_in==0) return ocean;
      else return elements[label_in - 1];
    }
}
