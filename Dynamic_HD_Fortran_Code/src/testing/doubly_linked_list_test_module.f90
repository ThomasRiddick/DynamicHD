module doubly_linked_list_test_module
use fruit
implicit none

contains

    subroutine setup
    end subroutine setup

    subroutine teardown
    end subroutine teardown

    subroutine testAddingReals
        use doubly_linked_list_mod
        type(doubly_linked_list) :: list
        real :: result1, result2, result3
        logical :: end0
            call list%add_value_to_back(1.0)
            call list%add_value_to_back(1.5)
            call list%add_value_to_front(-9.7)
            end0 = list%iterate_forward()
            select type(result => list%get_value_at_iterator_position())
            type is (real)
                result1 = result
            end select
            end0 = list%iterate_forward()
            select type(result => list%get_value_at_iterator_position())
            type is (real)
                result2 = result
            end select
            end0 = list%iterate_forward()
            select type(result => list%get_value_at_iterator_position())
            type is (real)
                result3 = result
            end select
            call assert_equals(-9.7,result1)
            call assert_equals(1.0,result2)
            call assert_equals(1.5,result3)
            call list%destructor()
            if (end0) continue
    end subroutine testAddingReals

    subroutine testAddingLogicals
        use doubly_linked_list_mod
        type(doubly_linked_list) :: list
        logical :: result1, result2, result3
        logical :: end0
            call list%add_value_to_back(.TRUE.)
            call list%add_value_to_back(.FALSE.)
            call list%add_value_to_front(.FALSE.)
            end0 = list%iterate_forward()
            select type(result => list%get_value_at_iterator_position())
            type is (logical)
                result1 = result
            end select
            end0 = list%iterate_forward()
            select type(result => list%get_value_at_iterator_position())
            type is (logical)
                result2 = result
            end select
            end0 = list%iterate_forward()
            select type(result => list%get_value_at_iterator_position())
            type is (logical)
                result3 = result
            end select
            call assert_equals(.FALSE.,result1)
            call assert_equals(.TRUE.,result2)
            call assert_equals(.FALSE.,result3)
            call list%destructor()
            if (end0) continue
    end subroutine testAddingLogicals

    subroutine testAddingCoords
        use doubly_linked_list_mod
        use coords_mod
        type(doubly_linked_list) :: list
        integer :: result1, result2, result3
        integer :: result4, result5, result6
        logical :: end0
            call list%add_value_to_back(latlon_coords(1,2))
            call list%add_value_to_back(latlon_coords(3,4))
            call list%add_value_to_front(latlon_coords(-5,-6))
            end0 = list%iterate_forward()
            select type(result => list%get_value_at_iterator_position())
            type is (latlon_coords)
                result1 = result%lat
                result2 = result%lon
            end select
            end0 = list%iterate_forward()
            select type(result => list%get_value_at_iterator_position())
            type is (latlon_coords)
                result3 = result%lat
                result4 = result%lon
            end select
            end0 = list%iterate_forward()
            select type(result => list%get_value_at_iterator_position())
            type is (latlon_coords)
                result5 = result%lat
                result6 = result%lon
            end select
            call assert_equals(-5,result1)
            call assert_equals(-6,result2)
            call assert_equals(1,result3)
            call assert_equals(2,result4)
            call assert_equals(3,result5)
            call assert_equals(4,result6)
            call list%destructor()
            if (end0) continue
    end subroutine testAddingCoords

    subroutine testAddingIntegersAndIteration
        use doubly_linked_list_mod
        type(doubly_linked_list) :: list
        logical :: end0,end1,end2,end3,end4,end5,end6
        integer :: result1,result2,result3,result4,result5
        integer :: result6,result7,result8,result9,result10
        integer :: result11
        integer :: length
            call list%add_value_to_back(1)
            call list%add_value_to_back(2)
            call list%add_value_to_front(16)
            call list%add_value_to_back(4)
            end0 = list%iterate_forward()
            select type(result => list%get_value_at_iterator_position())
            type is (integer)
                result1 = result
            end select
            end0 = list%iterate_forward()
            select type(result => list%get_value_at_iterator_position())
            type is (integer)
                result2 = result
            end select
            end0 = list%iterate_forward()
            select type(result => list%get_value_at_iterator_position())
            type is (integer)
                result3 = result
            end select
            end0 = list%iterate_backward()
            select type(result => list%get_value_at_iterator_position())
            type is (integer)
                result4 = result
            end select
            end1 = list%iterate_forward()
            select type(result => list%get_value_at_iterator_position())
            type is (integer)
                result5 = result
            end select
            end2 = list%iterate_forward()
            select type(result => list%get_value_at_iterator_position())
            type is (integer)
                result6 = result
            end select
            end3 = list%iterate_forward()
            select type(result => list%get_value_at_iterator_position())
            type is (integer)
                result7 = result
            end select
            end4 = list%iterate_backward()
            select type(result => list%get_value_at_iterator_position())
            type is (integer)
                result8 = result
            end select
            end0 = list%iterate_backward()
            end5 = list%iterate_backward()
            select type(result => list%get_value_at_iterator_position())
            type is (integer)
                result9 = result
            end select
            end6 = list%iterate_backward()
            select type(result => list%get_value_at_iterator_position())
            type is (integer)
                result10 = result
            end select
            end0 = list%iterate_forward()
            end0 = list%iterate_forward()
            end0 = list%iterate_forward()
            call list%reset_iterator()
            end0 = list%iterate_forward()
            select type(result => list%get_value_at_iterator_position())
            type is (integer)
                result11 = result
            end select
            length = list%get_length()
            call assert_equals(16,result1)
            call assert_equals(1,result2)
            call assert_equals(2,result3)
            call assert_equals(1,result4)
            call assert_equals(2,result5)
            call assert_equals(4,result6)
            call assert_equals(4,result7)
            call assert_equals(2,result8)
            call assert_equals(16,result9)
            call assert_equals(16,result10)
            call assert_equals(16,result11)
            call assert_equals(.FALSE.,end1)
            call assert_equals(.FALSE.,end2)
            call assert_equals(.TRUE.,end3)
            call assert_equals(.FALSE.,end4)
            call assert_equals(.FALSE.,end5)
            call assert_equals(.TRUE.,end6)
            call assert_equals(4,length)
            call list%destructor()
            if (end0) continue
    end subroutine testAddingIntegersAndIteration

    subroutine testRemoveElementAtIteratorPosition
        use doubly_linked_list_mod
        type(doubly_linked_list) :: list
        integer :: result1,result2,result3,result4,result5
        integer :: length0, length1, length2, length3
        logical :: end0, list_is_null
            call list%add_value_to_back(1)
            end0 = list%iterate_forward()
            call list%remove_element_at_iterator_position()
            list_is_null = .not. (associated(list%get_first_element_pointer()) .or. &
                                  associated(list%get_last_element_pointer()) .or. &
                                  associated(list%get_iterator_position()))
            call list%add_value_to_front(1)
            call list%add_value_to_back(2)
            call list%add_value_to_front(16)
            call list%add_value_to_back(4)
            call list%add_value_to_back(7)
            call list%add_value_to_back(6)
            call list%add_value_to_back(8)
            call list%add_value_to_front(15)
            length0 = list%get_length()
            end0 = list%iterate_forward()
            call list%remove_element_at_iterator_position()
            call list%reset_iterator()
            end0 = list%iterate_forward()
            select type(result => list%get_value_at_iterator_position())
            type is (integer)
                result1 = result
            end select
            length1 = list%get_length()
            end0 = list%iterate_forward()
            end0 = list%iterate_forward()
            end0 = list%iterate_forward()
            end0 = list%iterate_forward()
            end0 = list%iterate_forward()
            end0 = list%iterate_forward()
            end0 = list%iterate_forward()
            select type(result => list%get_value_at_iterator_position())
            type is (integer)
                result2 = result
            end select
            call list%remove_element_at_iterator_position()
            length2 = list%get_length()
            select type(result => list%get_value_at_iterator_position())
            type is (integer)
                result3 = result
            end select
            end0 = list%iterate_backward()
            end0 = list%iterate_backward()
            select type(result => list%get_value_at_iterator_position())
            type is (integer)
                result4 = result
            end select
            call list%remove_element_at_iterator_position()
            select type(result => list%get_value_at_iterator_position())
            type is (integer)
                result5 = result
            end select
            length3 = list%get_length()
            call assert_equals(.TRUE.,list_is_null)
            call assert_equals(16,result1)
            call assert_equals(8,result2)
            call assert_equals(6,result3)
            call assert_equals(4,result4)
            call assert_equals(2,result5)
            call assert_equals(8,length0)
            call assert_equals(7,length1)
            call assert_equals(6,length2)
            call assert_equals(5,length3)
            call list%destructor()
            if (end0) continue
    end subroutine testRemoveElementAtIteratorPosition

end module doubly_linked_list_test_module
