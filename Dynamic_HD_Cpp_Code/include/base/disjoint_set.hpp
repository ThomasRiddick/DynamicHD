/*
 * disjoint_set.hpp
 *
 *  Created on: Jun 14, 2020
 *      Author: thomasriddick
 */

// Does not include union by rank. Uses path splitting

#ifndef INCLUDE_DISJOINT_SET_HPP_
#define INCLUDE_DISJOINT_SET_HPP_

#include <vector>
using namespace std;

class disjoint_set {
  protected:
    int label;
    int size;
    disjoint_set* root = nullptr;
    vector<disjoint_set*>* nodes;
  public:
    disjoint_set(int label_in) : label(label_in), size(1) { root = this; }
    disjoint_set* get_root(){ return root;}
    void set_root(disjoint_set* x){ root = x; }
    void add_node(disjoint_set* x) { nodes->push_back(x); }
    void add_nodes(vector<disjoint_set*>* extra_nodes)
      { nodes->insert(nodes->end(),extra_nodes->begin(),
                      extra_nodes->end());}
    vector<disjoint_set*>* get_nodes() { return nodes; }
    void increase_size(int size_increment_in) {size = size+size_increment_in;}
    int get_size() { return size;}
    int get_label() { return label;}
};

class disjoint_sets{
  protected:
    vector<disjoint_set*> sets;
    vector<int> set_indices;
  public:
    disjoint_sets() {}
    disjoint_set* find_root(disjoint_set* x);
    int find_root(int label_in);
    bool link(disjoint_set* x,disjoint_set* y);
    bool make_new_link(int label_x, int label_y);
    void add_set(int label_in);
    disjoint_set* get_set(int label_in);
    void for_elements_in_set(disjoint_set* root,function<void(int)> func);
    void for_elements_in_set(int root_label,function<void(int)> func);
};

#endif //INCLUDE_DISJOINT_SET_HPP_
