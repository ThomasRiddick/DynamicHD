/*
 * priority_cell_queue.hpp
 *
 *  Created on: May 23, 2016
 *      Author: thomasriddick
 */

#ifndef INCLUDE_PRIORITY_CELL_QUEUE_HPP_
#define INCLUDE_PRIORITY_CELL_QUEUE_HPP_

#include "cell.hpp"
#include <queue>
using namespace std;

//Define a priority queue of cells
typedef priority_queue<cell*,vector<cell*>,compare_cells> priority_cell_queue_type;

/**
 * Very short class definition that inherits from the priority cell queue type then implements
 * a function called push that labels cells according to their order of entry.
 */

class priority_cell_queue : public priority_cell_queue_type {
private:
	//The number of entries that have been entered (including those entered and removed)
	//in the queue so far
	int k_count;

public:
	///Class constructor
	priority_cell_queue() : k_count(0) {};
	///Pushes a cell on the queue labeled by it order of entry (and updates the count)
	void push(cell* input) {input->set_k(k_count++); priority_cell_queue_type::push(input);}
	///Push a true sink labeled by the negation of the current count of entries
	///(and updates the count)
	void push_true_sink(cell* input) {
		input->set_k(-k_count++); priority_cell_queue_type::push(input);
	}
	///Get the next value of k to be used
	int get_next_k_value() {return k_count + 1;}
};

#endif /* INCLUDE_PRIORITY_CELL_QUEUE_HPP_ */
