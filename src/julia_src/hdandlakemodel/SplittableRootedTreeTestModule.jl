module RootedTreeTestModule

using Test: @test, @testset
using SplittableRootedTree: RootedTreeForest
using SplittableRootedTree: add_set,check_set_has_elements
using SplittableRootedTree: make_new_link,split_set
using SplittableRootedTree: get_all_node_labels_of_set
using SplittableRootedTree: find_root

@testset "Root tree tests" begin
  dsets::RootedTreeForest = RootedTreeForest()
  add_set(dsets,1)
  add_set(dsets,2)
  add_set(dsets,5)
  add_set(dsets,6)
  add_set(dsets,9)
  add_set(dsets,10)
  add_set(dsets,13)
  add_set(dsets,14)
  add_set(dsets,17)
  add_set(dsets,18)
  add_set(dsets,19)
  add_set(dsets,20)
  add_set(dsets,30)
  add_set(dsets,31)
  add_set(dsets,12)
  make_new_link(dsets,1,2)
  make_new_link(dsets,2,5)
  make_new_link(dsets,2,6)
  make_new_link(dsets,5,9)
  make_new_link(dsets,9,10)
  make_new_link(dsets,5,13)
  make_new_link(dsets,10,14)
  make_new_link(dsets,14,17)
  make_new_link(dsets,18,19)
  make_new_link(dsets,19,20)
  make_new_link(dsets,19,30)
  make_new_link(dsets,18,31)
  @test find_root(dsets,31) == 18
  @test find_root(dsets,18) == 18
  @test check_set_has_elements(dsets,1,Int64[1, 2, 5, 9, 10, 14, 17, 13, 6])
  @test check_set_has_elements(dsets,12,Int64[12])
  @test check_set_has_elements(dsets,18,Int64[18,19, 20, 30, 31])
  @test make_new_link(dsets,2,18)
  @test find_root(dsets,31) == 1
  @test find_root(dsets,18) == 1
  @test check_set_has_elements(dsets,12,Int64[12])
  @test check_set_has_elements(dsets,1,Int64[1, 2, 5, 9, 10, 14, 17, 13, 6, 18, 19, 20, 30, 31])
  @test (! make_new_link(dsets,10,1))
  @test split_set(dsets,18,19)
  @test check_set_has_elements(dsets,1,Int64[1, 2, 5, 9, 10, 14, 17, 13, 6, 18, 31])
  @test check_set_has_elements(dsets,19,Int64[19, 20, 30])
  @test split_set(dsets,1,18)
  @test find_root(dsets,31) == 18
  @test find_root(dsets,18) == 18
  @test check_set_has_elements(dsets,1,Int64[1, 2, 5, 9, 10, 14, 17, 13, 6])
  @test check_set_has_elements(dsets,18,Int64[18, 31])
  @test make_new_link(dsets,2,18)
  @test find_root(dsets,31) == 1
  @test find_root(dsets,18) == 1
  @test check_set_has_elements(dsets,1,Int64[1, 2, 5, 9, 10, 14, 17, 13, 6, 18, 31])
  @test check_set_has_elements(dsets,19,Int64[19, 20, 30])
  @test split_set(dsets,1,18)
  @test check_set_has_elements(dsets,1,Int64[1, 2, 5, 9, 10, 14, 17, 13, 6])
  @test check_set_has_elements(dsets,18,Int64[18, 31])
  @test find_root(dsets,31) == 18
  @test find_root(dsets,18) == 18
end

end
