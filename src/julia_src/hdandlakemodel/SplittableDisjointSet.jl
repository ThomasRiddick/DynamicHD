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
