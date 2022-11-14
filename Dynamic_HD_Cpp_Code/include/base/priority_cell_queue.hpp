/*
 * priority_cell_queue.hpp
 *
 *  Created on: May 23, 2016
 *      Author: thomasriddick
 */

#ifndef INCLUDE_PRIORITY_CELL_QUEUE_HPP_
#define INCLUDE_PRIORITY_CELL_QUEUE_HPP_

#include "base/cell.hpp"
#include <queue>
using namespace std;

//Define a priority queue of cells
typedef priority_queue<cell*,vector<cell*>,compare_cells> priority_cell_queue_type;

//Define a priority queue of cells in descending height order
typedef priority_queue<cell*,vector<cell*>,reverse_compare_cells> reverse_priority_cell_queue;

//Define a priority queue of reconnect type cells
typedef priority_queue<reconnect_cell*,vector<reconnect_cell*>,compare_reconnect_cells>
				priority_reconnect_cell_queue_type;

/**
 * Very short class definition that inherits from the priority cell queue type then implements
 * a function called push that labels cells according to their order of entry.
 */

template<class T,typename C>
class generic_priority_cell_queue : public T {
private:
	//The number of entries that have been entered (including those entered and removed)
	//in the queue so far
	int k_count;

public:
	///Class constructor
	generic_priority_cell_queue() : k_count(0) {};
	///Pushes a cell on the queue labeled by it order of entry (and updates the count)
	void push(C* input) {
			input->set_k(k_count++); T::push(input);
		#if PROCESSED_CELL_COUNTER
		if (k_count%1000000 == 0) cout << k_count/1000000 << "million" << endl;
		#endif
	}
	///Push a true sink labeled by the negation of the current count of entries
	///(and updates the count)
	void push_true_sink(C* input) {
		input->set_k(-k_count++); T::push(input);
	}
	///Get the next value of k to be used
	int get_next_k_value() {return k_count + 1;}
};

typedef generic_priority_cell_queue<priority_cell_queue_type,cell> priority_cell_queue;

typedef generic_priority_cell_queue<priority_reconnect_cell_queue_type,reconnect_cell>
				priority_reconnect_cell_queue;

#endif /* INCLUDE_PRIORITY_CELL_QUEUE_HPP_ */
