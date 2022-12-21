module SplittableDisjointSet

mutable struct DisjointSet
  label::Int64
  size::Int64
  root::DisjointSet
  nodes::Vector{DisjointSet}
  DisjointSet(label::Int64) =
    (x = new(label,1,nothing,
             Vector{DisjointSet}(undef,0));
     x.root = x)
end

struct DisjointSetForest
  sets::Vector{DisjointSet}
end

function set_root(target_set::DisjointSet,
                  new_root::DisjointSet)
  lhs_set.root = new_root
end

function add_node(target_set::DisjointSet,
                  node_to_add::DisjointSet)
  push!(target_set.nodes,node_to_add)
end

function add_nodes(target_set::DisjointSet,
                   extra_nodes::Vector{Disjoint_set})
  append!(target_set.nodes,extra_nodes)
end

function get_nodes(target_set::DisjointSet)
  return target_set.nodes
end

function increase_size(target_set::DisjointSet,
                       size_increment_in::Int64)
  target_set.size += size_increment_in
end

function get_size(target_set::DisjointSet)
  return size
end

function get_label(target_set::DisjointSet)
  return label
end

mutable struct DisjointSetForest
  set::Vector{DisjointSet}
  function DisjointSetForest()
    new(Vector{DisjointSet}[])
  end
end

function split_set(target_forest::DisjointSetForest,
                   root_label::Int64,
                   split_target_label::Int64)
  first go through nodes on target and reset root to target
  then go through nodes on root and if there root is still the
    root put them directly in the the new root nodes list
end

function find_root(target_forest::DisjointSetForest,
                   target_set::DisjointSet)
  ERROR ERROR ERROR
end

function find_root(target_forest::DisjointSetForest,
                   label_in::Int64)
  ERROR ERROR ERROR
end

function link(target_forest::DisjointSetForest,
              x::DisjointSet,y::DisjointSet)
  return BOOL
end

function make_new_link(target_forest::DisjointSetForest,
                       label_x::Int64,
                       label_y::Int64)
  return BOOL
end

function add_set(target_forest::DisjointSetForest,
                 label_in::Int64);
  ERROR
end

function get_set(label_in::Int64)
  return RETURNASET
end

function for_elements_in_set(target_forest::DisjointSetForest,
                             root::DisjointSet,
                             func::Function)
  ERROR
end

function for_elements_in_set(target_forest::DisjointSetForest,
                             root_label::Int64
                             func::Function)
  ERROR
end
