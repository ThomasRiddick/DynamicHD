/*
 * depression_hierarchy_construction_algorithm.cpp
 *
 *  Created on: Feb 10, 2020
 *      Author: thomasriddick
 * Implements the algorithm of Barnes et al 2020
 * Computing water flow through complex landscapes – Part 2: Finding hierarchies in depressions and morphological segmentations
 * Earth Surface Dynamics
 */


void depression_hierarchy_construction_algorithm::construct_hierarchy(){
  dh = disjoint_set(outlets->size());
  map<outlet_label<int,int>,double>::iterator iter = outlets->begin()
  while (iter != outlets.end()){
    element* first_depression_root = dh.find_root(dh.get_element(iter->first.first));
    element* second_depression_root = dh.find_root(dh.get_element(iter->first.second));
    if (first_depression_root != second_depression_root){
      dh.link(first_depression_root,second_depression_root);
    }
    iter++;
  }
}
