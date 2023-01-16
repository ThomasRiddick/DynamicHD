#include "base/disjoint_set.hpp"

using namespace std;

disjoint_set* disjoint_set_forest::find_root(disjoint_set* x){
  disjoint_set* root = x;
  disjoint_set* working_ptr;
  while (root->get_root() != root){
    working_ptr = root->get_root();
    working_ptr->set_root(root->get_root()->get_root());
    root = working_ptr;
  }
  return root;
}

bool disjoint_set_forest::link(disjoint_set* x, disjoint_set* y){
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

void disjoint_set_forest::add_set(int label_in){
  if (get_set(label_in)) return;
  disjoint_set* new_set = new disjoint_set(label_in);
  sets.push_back(new_set);
}

disjoint_set* disjoint_set_forest::get_set(int label_in){
  for (vector<disjoint_set*>::iterator i = sets.begin();i != sets.end(); ++i){
    if ((*i)->get_label() == label_in) return *i;
  }
  return nullptr;
}

bool disjoint_set_forest::make_new_link(int label_x, int label_y){
  disjoint_set* x = get_set(label_x);
  disjoint_set* y = get_set(label_y);
  return link(x,y);
}

int disjoint_set_forest::find_root(int label_in){
  disjoint_set* x = get_set(label_in);
  disjoint_set* root_x = find_root(x);
  return root_x->get_label();
}

void disjoint_set_forest::for_elements_in_set(disjoint_set* root,function<void(int)> func){
  if (root->get_root() != root)
    throw runtime_error("Given set is not label of a root set");
  func(root->get_label());
  for (vector<disjoint_set*>::iterator i = root->get_nodes()->begin();
       i != root->get_nodes()->end(); ++i){
    func((*i)->get_label());
  }
}

void disjoint_set_forest::for_elements_in_set(int root_label,function<void(int)> func){
  disjoint_set* root = get_set(root_label);
  for_elements_in_set(root,func);
}

bool disjoint_set_forest::check_subset_has_elements(int label_of_element,
                                              vector<int> element_labels){
  bool check_passed = true;
  disjoint_set* root = find_root(get_set(label_of_element));
  if (root->get_nodes()->size() + 1 != element_labels.size()) return false;
  if (element_labels[0] != root->get_label()) return false;
  int j = 1;
  for (vector<disjoint_set*>::iterator i = root->get_nodes()->begin();
       i != root->get_nodes()->end(); ++i){
    if (element_labels[j] != (*i)->get_label()) check_passed = false;
    j++;
  }
  return check_passed;
}

disjoint_set_forest::~disjoint_set_forest(){
  for (vector<disjoint_set*>::iterator i = sets.begin();
       i != sets.end(); ++i){
       delete (*i);
  }
}

ostream& operator<<(ostream& out, disjoint_set_forest& sets_object){
  for (vector<disjoint_set*>::iterator i = sets_object.sets.begin();
       i != sets_object.sets.end(); ++i){
      int label = (*i)->get_label();
      if (label == sets_object.find_root(label)){
        out << label << ":";
        sets_object.for_elements_in_set(label,[&](int subset_label){
          out << subset_label << ", ";
        });
        out << endl;
      }
  }
  return out;
}

