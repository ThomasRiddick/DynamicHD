class Lake:

    def __init__(self,
                 lake_number,
                 center_coords,
                 is_leaf=True,
                 primary_lake=None,
                 secondary_lakes=None):
        #Keep references to lakes as number not objects
        self.center_coords = center_coords
        self.lake_number = lake_number
        self.is_leaf = is_leaf
        self.primary_lake = primary_lake
        self.secondary_lakes  = secondary_lakes
        self.sill_point = None

    def set_sill_point(self,coords):
        self.sill_point = coords

    def set_primary_lake(self,primary_lake)
        self.primary_lake = primary_lake

class BasinCell:

    self __init__(cell_coords,cell_height_type,cell_height):
        self.cell_coords = center_coords
        self.cell_height_type = cell_height_type
        self.cell_height = cell_height

class

class disjoint_set {
  protected:
    int label;
    int size;
    disjoint_set* root = nullptr;
    vector<disjoint_set*>* nodes = nullptr;
  public:
    disjoint_set(int label_in) : label(label_in), size(1)
      { root = this; nodes = new vector<disjoint_set*>(); }
    ~disjoint_set() { delete nodes; }
    disjoint_set* get_root(){ return root;}
    void set_root(disjoint_set* x){ root = x; }
    void add_node(disjoint_set* x) { nodes->push_back(x); }
    void add_nodes(vector<disjoint_set*>* extra_nodes)
      { nodes->insert(nodes->end(),extra_nodes->begin(),
                      extra_nodes->end());}
    vector<disjoint_set*>* get_nodes() { return nodes; }
    void increase_size(int size_increment_in) {size += size_increment_in;}
    int get_size() { return size;}
    int get_label() { return label;}
    friend ostream& operator<<(ostream& out, disjoint_set& set_object)
    { return out << set_object.get_label(); }
};

class disjoint_set_forest{
  protected:
    vector<disjoint_set*> sets;
  public:
    disjoint_set_forest() {}
    ~disjoint_set_forest();
    disjoint_set* find_root(disjoint_set* x);
    int find_root(int label_in);
    bool link(disjoint_set* x,disjoint_set* y);
    bool make_new_link(int label_x, int label_y);
    void add_set(int label_in);
    disjoint_set* get_set(int label_in);
    void for_elements_in_set(disjoint_set* root,function<void(int)> func);
    void for_elements_in_set(int root_label,function<void(int)> func);
    bool check_subset_has_elements(int label_of_element,vector<int> element_labels);
    friend ostream& operator<<(ostream& out, disjoint_set_forest& sets_object);
};

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
  root_y->set_root(root_x);
  root_x->increase_size(root_y->get_size());
  root_x->add_node(root_y);
  root_x->add_nodes(root_y->get_nodes());
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


class BasinEvaluationAlgorithm:

    def evaluate_basins(self):
        while not self.mimima.empty():
            minimum = self.minima.top()
            self.minima.heappop()
            lake_number = len(self.lakes)
            self.lakes.append(Lake(lake_number,minimum))
            self.lake_q.append(lake_number)
            self.lake_connections.add_set(lake_number)
        while not self.lake_q.empty():
            lake = lake_q.top()
            lake_q.pop()
            self.initialize_basin(lake)
            while True:
                if self.q.empty():
                    raise RuntimeError("Basin outflow not found")
                self.center_cell = self.q.top()
                self.q.heappop()
                #Call the newly loaded coordinates and height for the center cell 'new'
                #until after making the test for merges then relabel. Center cell height/coords
                #without the 'new' moniker refers to the previous center cell; previous center cell
                #height/coords the previous previous center cell
                new_center_coords = self.center_cell.get_cell_coords().clone()
                new_center_cell_height_type = self.center_cell.get_height_type()
                new_center_cell_height = self.center_cell.get_height()
                #Exit to basin or level area found
                if (new_center_cell_height <= self.center_cell_height and
                    searched_level_height != self.center_cell_height):
                        outflow_coords_list = self.search_for_outflows_on_level(self.q
                                                                                self.center_coords,
                                                                                self.center_cell_height)
                        if len(outflow_coords_list) > 0:
                            #Exit(s) found
                            for outflow_coords in outflow_coords_list
                                other_lake_number = lake_numbers[outflow_coords]
                                if other_lake_number != -1
                                    self.lake_connections.make_new_link(lake_number,other_basin_number)
                                    merging_lakes.append(lake_number)
                                    #The unusual case of two outflows on seperate disconnected
                                    #plateaus will not be handled correctly but this will not
                                    #cause an error
                                    self.lake.set_sill_point(self.center_coords)
                            self.fill_lake_orography(self.lake)
                            break
                        else:
                            #Don't rescan level later
                            searched_level_height = self.center_cell_height
                #Process neighbors of new center coords
                process_neighbors()
                previous_filled_cell_coords = center_coords.clone()
                previous_filled_cell_height = self.center_cell_height
                previous_filled_cell_height_type = self.center_cell_height_type
                center_cell_height_type = new_center_cell_height_type
                center_cell_height = new_center_cell_height
                center_coords = new_center_coords
                process_center_cell()
            if len(merging_lakes) > 0:
                unique_lake_groups = {g for g in [self.lake_connections.find_root(l)
                                                  for l in merging_lakes]}
                for lake_group in unique_lake_groups:
                    sublakes_in_lake = [sublake for sublake in self.lake_connections.get_set(lake_group)
                                        if sublake.primary_lake is None ]
                    new_lake_number = len(lakes)
                    new_lake = Lake(new_lake_number,
                                    lakes[sublakes_in_lake[0]].sill_point,
                                    is_leaf=False,
                                    primary_lake=None,
                                    secondary_lakes=sublakes_in_lake)
                    #Note the new lake isn't necessarily the root of the disjointed set
                    lake_connections.make_new_link(new_lake_number)
                    lakes.append(new_lake)
                    for sublake in sublakes_in_lake:
                        lake_numbers[lake_numbers == sublake] = new_lake.lake_number
                        lakes[sublake].set_primary_lake(new_lake.lake_number)
                merging_lakes = []
            else:
                break

    def initialize_basin(self.lake):
        self.completed_cells[:,:] = False
        self.center_cell_volume_threshold = 0.0
        self.center_coords = lake.center_coords.clone()
        raw_height = raw_orography[self.center_coords]
        corrected_height = corrected_orography[self.center_coords]
        if raw_height <= corrected_height:
            self.center_cell_height = raw_height
            self.center_cell_height_type = flood_height
        else:
            self.center_cell_height = corrected_height
            self.center_cell_height_type = connection_height
        self.previous_filled_cell_coords = center_coords.clone()
        self.previous_filled_cell_height_type = center_cell_height_type
        self.previous_filled_cell_height = center_cell_height
        self.catchments_from_sink_filling_catchment_num =
            catchments_from_sink_filling[center_coords]
        new_center_coords = center_coords.clone()
        new_center_cell_height_type = center_cell_height_type
        new_center_cell_height = center_cell_height
        if self.center_cell_height_type == connection_height:
            self.lake_area = 0.0
        else if self.center_cell_height_type == flood_height:
            self.lake_area = cell_areas[center_coords]
        else:
            raise RuntimeError("Cell type not recognized")
        self.completed_cells[center_coords] = True
        #Make partial first and second iteration
        self.process_neighbors()
        self.center_cell = self.q.top()
        self.q.heappop()
        new_center_coords = self.center_cell.get_cell_coords().clone()
        new_center_cell_height_type = self.center_cell.get_height_type()
        new_center_cell_height = self.center_cell.get_height()
        center_cell_height_type = new_center_cell_height_type
        center_cell_height = new_center_cell_height
        center_coords = new_center_coords
        self.process_center_cell()

    def search_for_outflows_on_level(self,q
                                     center_cell,
                                     center_cell_height):
        level_q.push_back(center_cell)
        while self.q[0].get_height() == center_cell_height:
            level_q.push_back(self.q.heappop())
        for cell in level_q:
            self.q.heappush(cell)
        while not level_q.empty():
            level_center_cell = level_q.pop()
            self.process_level_neighbors(level_center_cell.get_coords())
        return self.outflows_on_level

    def process_level_neighbors(self,level_coords):
        neighbors_coords = raw_orography.get_neighbors_coords(level_coords,1)
        while not neighbors_coords.empty():
            nbr_coords = neighbors_coords.back()
            neighbors_coords.pop_back()
            if (not level_completed_cells[nbr_coords] and
                not self.completed_cells[nbr_coords]):
                raw_height = raw_orography[nbr_coords]
                corrected_height = corrected_orography[nbr_coords]
                level_completed_cells[nbr_coords] = True
                if raw_height <= corrected_height:
                    nbr_height = raw_height
                    nbr_height_type = flood_height
                else:
                    nbr_height = corrected_height
                    nbr_height_type = connection_height
                if nbr_height == self.center_cell_height:
                    level_q.push(BasinCell(nbr_height,nbr_height_type,
                                           nbr_coords))
                else if nbr_height < self.center_cell_height:
                    self.outflows_on_level.push_back(nbr_coords)


    def process_center_cell():
        if (basin_numbers[previous_filled_cell_coords] == null_catchment):
            basin_numbers[previous_filled_cell_coords] = basin_number
        center_cell_volume_threshold +=
                lake_area*(center_cell_height-previous_filled_cell_height)
        if previous_filled_cell_height_type == connection_height:
            self.q.heappush(BasinCell(raw_orography[previous_filled_cell_coords],
                                      flood_height,previous_filled_cell_coords.clone()))
            self.filling_order.append((coords_in,flood_height,center_cell_volume_threshold,
                                       center_cell_height))
        else if previous_filled_cell_height_type == flood_height:
            self.filling_order.append((coords_in,connect_height,center_cell_volume_threshold,
                                       center_cell_height))
        else:
            raise RuntimeError("Cell type not recognized")
        if (center_cell_height_type == flood_height):
            lake_area += cell_areas[center_coords]
        else if center_cell_height_type != connection_height:
            raise RuntimeError("Cell type not recognized")

    def process_neighbors(self):
        neighbors_coords = raw_orography.get_neighbors_coords(new_center_coords,1)
        while not neighbors_coords.empty():
            nbr_coords = neighbors_coords.back()
            neighbors_coords.pop_back()
            nbr_catchment = catchments_from_sink_filling[nbr_coords]
            in_different_catchment =
              ( nbr_catchment != catchments_from_sink_filling_catchment_num) and
              ( nbr_catchment != -1)
            if (not self.completed_cells[nbr_coords] and
                not in_different_catchment):
                raw_height = raw_orography[nbr_coords]
                corrected_height = corrected_orography[nbr_coords]
                if raw_height <= corrected_height:
                    nbr_height = raw_height
                    nbr_height_type = flood_height
                else:
                    nbr_height = corrected_height
                    nbr_height_type = connection_height
                self.q.heappush(BasinCell(nbr_height,nbr_height_type,
                                          nbr_coords))
                self.completed_cells[nbr_coords] = True
