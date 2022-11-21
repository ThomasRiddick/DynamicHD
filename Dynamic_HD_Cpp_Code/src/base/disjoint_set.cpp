#include "disjoint_set.hpp"

using namespace std;

void disjoint_sets::find_root(set* x){
  set* root = x;
  set* working_ptr;
  while (root->get_root()->get_root() != root->get_root()){
    working_ptr = root->get_root();
    root->get_root() = root->get_root()->get_root();
    root = working_ptr;
  }
  return root;
}

bool disjoint_sets::link(set* x, set* y){
  set* root_x find_root(x);
  set* root_y find_root(y);
  if (root_x == root_y) return false;
  if (root_x.get_size() < root_y.get_size()) {
    y->set_root(x);
    y->increase_size(x->get_size());
  } else {
    x->set_root(y)
    x->increase_size(y->get_size());
  }
  return true;
}

void disjoint_sets::add_set(label_in){
  if (! get_set(label_in)) return;
  set* new_set = new set();
  sets->push_back(new_set);
}

set* disjoint_sets::get_set(label_in){
  for (vector<set*>::iterator j = i->sets.begin();j != i->sets.end(); ++j){
    if (j->get_label() == label_in) return j;
  }
  return nullptr;
}

bool disjoint_sets::make_new_link(int label_x, int label_y){
  set* x = get_set(label_x);
  set* y = get_set(label_y);
  return link(x,y);
}
