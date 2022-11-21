#include "disjoint_set.hpp"

using namespace std;

void disjoint_set::find_root(element* x){
  if (x->get_root()->get_root() == x->get_root()) return x->get_root()
  element* root = x;
  while (root->get_root() != root){
    root = root->get_root();
  }

  while (x->get_root() != root){
    element* parent = x->get_root();
    x->set_root(root);
    x = parent
  }
  return root;
}

void disjoint_set::link(element* x, element* y){
  element* root_x find_root(x);
  element* root_y find_root(y);
  element* z = add_element();
  root_x->set_parent(z);
  root_x->set_root(z);
  root_y->set_parent(z);
  root_y->set_root(z);
}

element* disjoint_set::add_element(){
  num_elements++;
  element* new_element = new element(num_elements);
  elements->push_back(new_element);
}
