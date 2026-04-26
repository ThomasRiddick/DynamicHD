void accumulate_flow_icon_single_index(int ncells,
                                       int* neighboring_cell_indices_in,
                                       int* input_next_cell_index,
                                       int* output_cumulative_flow,
                                       int* bifurcated_next_cell_index = nullptr);

void accumulate_flow_latlon(int nlat, int nlon,
                            int* input_river_directions,
                            int* output_cumulative_flow);
