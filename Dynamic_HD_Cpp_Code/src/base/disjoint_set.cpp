#include "base/disjoint_set.hpp"

using namespace std;

disjoint_set* disjoint_sets::find_root(disjoint_set* x){
  disjoint_set* root = x;
  disjoint_set* root_of_root;
  disjoint_set* working_ptr;
  while (root->get_root()->get_root() != root->get_root()){
    working_ptr = root->get_root();
    root_of_root = root->get_root();
    root_of_root = root->get_root()->get_root();
    root = working_ptr;
  }
  return root;
}

bool disjoint_sets::link(disjoint_set* x, disjoint_set* y){
  disjoint_set* root_x = find_root(x);
  disjoint_set* root_y = find_root(y);
  if (root_x == root_y) return false;
  if (root_x->get_size() < root_y->get_size()) {
    root_x->set_root(root_y);
    root_y->increase_size(root_x->get_size());
    root_y->add_node(root_x);
    root_y->add_nodes(root_x->get_nodes());
  } else {
    root_y->set_root(root_x);
    root_x->increase_size(root_y->get_size());
    root_x->add_node(root_y);
    root_x->add_nodes(root_y->get_nodes());
  }
  return true;
}

void disjoint_sets::add_set(int label_in){
  if (! get_set(label_in)) return;
  disjoint_set* new_set = new disjoint_set(label_in);
  sets.push_back(new_set);
}

disjoint_set* disjoint_sets::get_set(int label_in){
  for (vector<disjoint_set*>::iterator i = sets.begin();i != sets.end(); ++i){
    if ((*i)->get_label() == label_in) return *i;
  }
  return nullptr;
}

bool disjoint_sets::make_new_link(int label_x, int label_y){
  disjoint_set* x = get_set(label_x);
  disjoint_set* y = get_set(label_y);
  return link(x,y);
}

int disjoint_sets::find_root(int label_in){
  disjoint_set* x = get_set(label_in);
  disjoint_set* root_x = find_root(x);
  return root_x->get_label();
}

void disjoint_sets::for_elements_in_set(disjoint_set* root,function<void(int)> func){
  func(root->get_label());
  for (vector<disjoint_set*>::iterator i = root->get_nodes()->begin();
       i != root->get_nodes()->end(); ++i){
    func((*i)->get_label());
  }
}

void disjoint_sets::for_elements_in_set(int root_label,function<void(int)> func){
  disjoint_set* root = get_set(root_label);
  for_elements_in_set(root,func);
}


