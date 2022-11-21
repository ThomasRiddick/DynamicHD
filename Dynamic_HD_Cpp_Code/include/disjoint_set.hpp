/*
 * disjoint_set.hpp
 *
 *  Created on: Jun 14, 2020
 *      Author: thomasriddick
 */

// Does not include union by rank. Uses path splitting

using namespace std;

class element {
  protected:
    int label;
    element* parent = nullptr;
    element* root = nullptr;
  public:
    element(int label_in) : label(label_in) {parent = this; root = this}
    element* get_parent(){ return parent;}
    element* get_root(){ return root;}
    void set_parent(element* x){ parent = x;}
    void set_root(element* x){ root = x; }
}

class disjoint_set{
  protected:
    num_elements;
    vector<element*> elements;
  public:
    disjoint_set() : num_elements(0) {}
    disjoint_set(num_elements_in) {
      num_elements = 0;
      for (int i=0;i<num_elements_in;i++){
        add_element();
      }
    }
    element* find_root(element* x);
    void link(element* x,element* y);
    element* add_element();
    element* get_element(int label_in){ return elements[label_in - 1];}
}
