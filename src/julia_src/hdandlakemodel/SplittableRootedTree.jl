module SplittableRootedTree

import Base.show

mutable struct RootedTree
  label::Int64
  direct_nodes::Vector{RootedTree}
  root::RootedTree
  superior::RootedTree
  RootedTree(label::Int64) =
    (x = new(label,RootedTree[]);
     x.root = x; x.superior = x)
end

function set_root(target_set::RootedTree,
                  new_root::RootedTree)
  target_set.root = new_root
end

function set_superior(target_set::RootedTree,
                      new_superior::RootedTree)
  target_set.superior = new_superior
end

function add_direct_node(target_set::RootedTree,
                         node_to_add::RootedTree)
  push!(target_set.direct_nodes,node_to_add)
end

function get_direct_nodes(target_set::RootedTree)
  return target_set.direct_nodes
end

function get_label(target_set::RootedTree)
  return target_set.label
end

function show(io::IO,set::RootedTree)
  println("label: $(set.label)")
  print(" direct nodes:")
  for set in set.direct_nodes
    print("$(set.label) ")
  end
  println()
end

function find_root(target_set::RootedTree)
  root::RootedTree = target_set
  while root.root != root
    working_ptr::RootedTree = root.root
    set_root(working_ptr,root.root.root)
    root = working_ptr
  end
  return root
end

function link(x::RootedTree,y::RootedTree)
  root_x::RootedTree = find_root(x)
  if y.root != y
    error("rhs set must be tree root when adding link")
  end
  if root_x == y
    return false
  end
  set_root(y,root_x)
  add_direct_node(x,y)
  set_superior(y,x)
  return true
end

struct RootedTreeForest
  sets::Vector{RootedTree}
  function RootedTreeForest()
    new(RootedTree[])
  end
end

function find_root(target_forest::RootedTreeForest,
                   label_in::Int64)
  x::RootedTree = get_set(target_forest,label_in)
  root_x::RootedTree = find_root(x)
  return root_x.label::Int64
end

function make_new_link(target_forest::RootedTreeForest,
                       label_x::Int64,
                       label_y::Int64)
  x::RootedTree = get_set(target_forest,label_x)
  y::RootedTree = get_set(target_forest,label_y)
  return link(x,y)
end

function split_set(target_forest::RootedTreeForest,
                   set_label::Int64,
                   subset_to_split_label::Int64)
  original_root::RootedTree = find_root(get_set(target_forest,
                                                set_label))
  new_subset_root::RootedTree = get_set(target_forest,subset_to_split_label)
  if find_root(new_subset_root) != original_root
    return false
  end
  set_root(new_subset_root,new_subset_root)
  for_elements_in_set(target_forest,new_subset_root,x->set_root(x,new_subset_root))
  filter!(x->(x!=new_subset_root),new_subset_root.superior.direct_nodes)
  set_superior(new_subset_root,new_subset_root)
  return true
end

function add_set(target_forest::RootedTreeForest,
                 label_in::Int64)
  if contains_set(target_forest,label_in)
    return
  end
  new_set::RootedTree = RootedTree(label_in)
  push!(target_forest.sets,new_set)
end

function contains_set(target_forest::RootedTreeForest,
                      label_in::Int64)
  for i in target_forest.sets
    if get_label(i) == label_in
      return true
    end
  end
  return false
end

function get_set(target_forest::RootedTreeForest,
                 label_in::Int64)
  for i in target_forest.sets
    if get_label(i) == label_in
      return i::RootedTree
    end
  end
  error("Requested set doesn't exist")
end

function for_elements_in_set(target_forest::RootedTreeForest,
                             element::RootedTree,
                             func::Function)
  func(element)
  for i in get_direct_nodes(element)
    for_elements_in_set(target_forest,i,func)
  end
end

function for_elements_in_set(target_forest::RootedTreeForest,
                             element_label::Int64,
                             func::Function)
  element::RootedTree = get_set(target_forest,element_label)
  for_elements_in_set(target_forest,element,func)
end

function check_set_has_elements(target_forest::RootedTreeForest,
                                label_of_element::Int64,
                                element_labels::Vector{Int64})
  node_labels::Vector{Int64} =
    get_all_node_labels_of_set(target_forest,
                               label_of_element)
  if length(node_labels) != length(element_labels)
    return false
  end
  return all(node_labels .== element_labels)
end

function get_all_node_labels_of_set(target_forest::RootedTreeForest,
                                    label_of_element::Int64)
  root::RootedTree = find_root(get_set(target_forest,
                                        label_of_element))
  full_node_list::Vector{Int64} = Int64[]
  for_elements_in_set(target_forest,root,
                      x->push!(full_node_list,x.label))
  return full_node_list
end


function show(io::IO,forest::RootedTreeForest)
  for set in forest.sets
    println(set)
  end
end

end
