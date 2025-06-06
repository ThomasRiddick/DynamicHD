module lake_model_tree_test_mod
use fruit
implicit none

contains

    subroutine setup
    end subroutine setup

    subroutine teardown
    end subroutine teardown

subroutine testRootedTrees
   use lake_model_tree_mod
   type(rooted_tree_forest), pointer :: dsets
   integer, dimension(:), pointer :: elements
   logical :: dummy
    dsets => rooted_tree_forest()
    call dsets%add_set(1)
    call dsets%add_set(2)
    call dsets%add_set(5)
    call dsets%add_set(6)
    call dsets%add_set(9)
    call dsets%add_set(10)
    call dsets%add_set(13)
    call dsets%add_set(14)
    call dsets%add_set(17)
    call dsets%add_set(18)
    call dsets%add_set(19)
    call dsets%add_set(20)
    call dsets%add_set(30)
    call dsets%add_set(31)
    call dsets%add_set(12)
    dummy = dsets%make_new_link_from_labels(1,2)
    dummy = dsets%make_new_link_from_labels(2,5)
    dummy = dsets%make_new_link_from_labels(2,6)
    dummy = dsets%make_new_link_from_labels(5,9)
    dummy = dsets%make_new_link_from_labels(9,10)
    dummy = dsets%make_new_link_from_labels(5,13)
    dummy = dsets%make_new_link_from_labels(10,14)
    dummy = dsets%make_new_link_from_labels(14,17)
    dummy = dsets%make_new_link_from_labels(18,19)
    dummy = dsets%make_new_link_from_labels(19,20)
    dummy = dsets%make_new_link_from_labels(19,30)
    dummy = dsets%make_new_link_from_labels(18,31)
    allocate(elements(9))
    elements = (/ 1, 2, 5, 6, 9, 10, 13, 14, 17 /)
    call assert_true(dsets%check_set_has_elements(1,elements))
    deallocate(elements)
    allocate(elements(1))
    elements = (/ 12 /)
    call assert_true(dsets%check_set_has_elements(12,elements))
    deallocate(elements)
    allocate(elements(5))
    elements = (/ 18,19, 20, 30, 31 /)
    call assert_true(dsets%check_set_has_elements(18,elements))
    deallocate(elements)
    call assert_true(dsets%make_new_link_from_labels(2,18))
    allocate(elements(1))
    elements = (/ 12 /)
    call assert_true(dsets%check_set_has_elements(12,elements))
    deallocate(elements)
    allocate(elements(14))
    elements = (/ 1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 19, 20, 30, 31 /)
    call assert_true(dsets%check_set_has_elements(1,elements))
    deallocate(elements)
    call assert_false(dsets%make_new_link_from_labels(10,1))
    call assert_true(dsets%split_set(18,19))
    allocate(elements(11))
    elements = (/ 1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 31 /)
    call assert_true(dsets%check_set_has_elements(1,elements))
    deallocate(elements)
    allocate(elements(3))
    elements = (/ 19, 20, 30 /)
    call assert_true(dsets%check_set_has_elements(19,elements))
    deallocate(elements)
    call assert_true(dsets%split_set(1,18))
    allocate(elements(9))
    elements = (/ 1, 2, 5, 6, 9, 10, 13, 14, 17 /)
    call assert_true(dsets%check_set_has_elements(1,elements))
    deallocate(elements)
    allocate(elements(2))
    elements = (/ 18, 31 /)
    call assert_true(dsets%check_set_has_elements(18,elements))
    deallocate(elements)
    call assert_true(dsets%make_new_link_from_labels(2,18))
    allocate(elements(11))
    elements = (/ 1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 31 /)
    call assert_true(dsets%check_set_has_elements(1,elements))
    deallocate(elements)
    allocate(elements(3))
    elements = (/ 19, 20, 30 /)
    call assert_true(dsets%check_set_has_elements(19,elements))
    deallocate(elements)
    call assert_true(dsets%split_set(1,18))
    allocate(elements(9))
    elements = (/ 1, 2, 5, 6, 9, 10, 13, 14, 17 /)
    call assert_true(dsets%check_set_has_elements(1,elements))
    deallocate(elements)
    allocate(elements(2))
    elements = (/ 18, 31 /)
    call assert_true(dsets%check_set_has_elements(18,elements))
    deallocate(elements)
    call dsets%rooted_tree_forest_destructor()
    deallocate(dsets)
end subroutine testRootedTrees

end module lake_model_tree_test_mod
