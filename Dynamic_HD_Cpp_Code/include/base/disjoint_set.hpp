/*
 * disjoint_set.hpp
 *
 *  Created on: Jun 14, 2020
 *      Author: thomasriddick
 */

// Does not include union by rank. Uses path splitting

using namespace std;

class set {
  protected:
    int label;
    int size;
    set* root = nullptr;
  public:
    set(int label_in) : label(label_in), size(1) {parent = this; root = this}
    set* get_root(){ return root;}
    void set_root(set* x){ root = x; }
    void increase_size(int size_increment_in) {size = size+size_increment_in;}
    int get_size() { return size;}
}

class disjoint_sets{
  protected:
    vector<set*> sets;
    vector<int> set_indices;
  public:
    disjoint_sets() {}
    set* find_root(set* x);
    bool link(set* x,set* y);
    bool make_new_link(int label_x, int label_y);
    set* add_set(int label_in);
    set* get_set(int label_in);
}
