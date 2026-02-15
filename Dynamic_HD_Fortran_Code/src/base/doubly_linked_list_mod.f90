module doubly_linked_list_mod
!For gfortran 7.2.0
!For reasons unknown constructor function overloading is not working properly in gfortran
!hence alias the full name of the constructor to create_link
!use doubly_linked_list_link_mod,  create_link => doubly_linked_list_link_constructor
use doubly_linked_list_link_mod
use coords_mod
implicit none
private

!> Class that implements a doubly linked list (a list that can be iterated both backwards
!! and forwards)
type, public :: doubly_linked_list
    private
    !> Pointer to the first link in the list
    type(doubly_linked_list_link), pointer :: first_element => null()
    !> Pointer to the last link in the list
    type(doubly_linked_list_link), pointer :: last_element => null()
    !> Pointer to the link which the iterator is currently positioned at
    type(doubly_linked_list_link), pointer :: iterator_position => null()
    !> Current length of this list
    integer :: length
    contains
        !> Class destructor. If the list still has links this iterates through
        !! them and removes and deletes them
        procedure :: destructor
        !> Getter for length
        procedure :: get_length
        !> Add a value of a generic type to the back of the list
        procedure :: add_generic_value_to_back
        !> Add a value of a generic type to the front of the list
        procedure :: add_generic_value_to_front
        !> Generic type bound procedure for adding values to the back of the list;
        !! this then calls one of the type specific wrappers for adding a value to
        !! the back
        generic :: add_value_to_back => add_integer_to_back, add_real_to_back, &
            add_logical_to_back, add_coords_to_back
        !> Generic type bound procedure for adding value to the front of the list;
        !! this then calls one of the type specific wrappers for adding a value to
        !! the front
        generic :: add_value_to_front => add_integer_to_front, add_real_to_front, &
            add_logical_to_front, add_coords_to_front
        !> Wrapper to add an integer to the back of the list
        procedure :: add_integer_to_back
        !> Wrapper to add a real to the back of the list
        procedure :: add_real_to_back
        !> Wrapper to add a logical to the back of the list
        procedure :: add_logical_to_back
        !> Wrapper to add a coords object to the back of the list
        procedure :: add_coords_to_back
        !> Wrapper to add an integer to the front of the list
        procedure :: add_integer_to_front
        !> Wrapper to add a real to the front of the list
        procedure :: add_real_to_front
        !> Wrapper to add a logical to the front of the list
        procedure :: add_logical_to_front
        !> Wrapper to add a coords object to the front of the list
        procedure :: add_coords_to_front
        !> Move the iterator one position forward in the list; this is a
        !! function and returns a boolean flag to indicate if the end of
        !! the list has been reached (TRUE) or not (FALSE)
        procedure :: iterate_forward
        !> Move the iterator one position backward in the list; this is a
        !! function and returns a boolean flag to indicate if the start of
        !! the list has been reached (TRUE) or not (FALSE)
        procedure :: iterate_backward
        !> Fuction to get (a pointer to) the value located at the current
        !! position of the iterator
        procedure :: get_value_at_iterator_position
        !> Remove the element/link (and its value) at the current iterator
        !! position and join the gap created in the list (unless this is
        !! the end and/or start of the list)
        procedure :: remove_element_at_iterator_position
        !> Reset the iterator to a null position
        procedure :: reset_iterator
        !> Getter for pointer to last element
        procedure :: get_last_element_pointer
        !> Getter for pointer to first element
        procedure :: get_first_element_pointer
        !> Getter for pointer to position of the iterator
        procedure :: get_iterator_position
end type doubly_linked_list

interface doubly_linked_list
    procedure :: doubly_linked_list_constructor
end interface doubly_linked_list

contains

    function doubly_linked_list_constructor() result(constructor)
        type(doubly_linked_list), pointer :: constructor
            allocate(constructor)
            constructor%length = 0
    end function doubly_linked_list_constructor

    subroutine destructor(this)
        class(doubly_linked_list), intent(inout) :: this
            if (this%get_length() > 0) then
                call this%reset_iterator()
                do
                    if (this%iterate_forward()) exit
                    call this%remove_element_at_iterator_position()
                end do
            end if
    end subroutine destructor

    pure function get_length(this) result(length)
        class(doubly_linked_list), intent(in) :: this
        integer :: length
            length = this%length
    end function get_length

    subroutine add_generic_value_to_back(this,value)
        class(doubly_linked_list), intent(inout) :: this
        class(*), pointer, intent(inout) :: value
        type(doubly_linked_list_link), pointer :: new_element
        if (associated(this%first_element)) then
           this%length = this%length + 1
           !For gfortran 7.2.0
           !For reasons unknown constructor function overloading is not working properly in gfortran
           !new_element => create_link(value,null(this%first_element),this%last_element)
           new_element => doubly_linked_list_link(value,null(this%first_element),this%last_element)
           call this%last_element%set_next_element(new_element)
           this%last_element => new_element
        else
           this%length = 1
           !For gfortran 7.2.0
           !For reasons unknown constructor function overloading is not working properly in gfortran
           !this%first_element => create_link(value,null(),null())
           this%first_element => doubly_linked_list_link(value,null(),null())
           this%last_element => this%first_element
        end if
        deallocate(value)
    end subroutine add_generic_value_to_back

    subroutine add_generic_value_to_front(this,value)
        class(doubly_linked_list), intent(inout) :: this
        class(*), pointer, intent(inout) :: value
        type(doubly_linked_list_link), pointer :: new_element
        if (associated(this%first_element)) then
            this%length = this%length + 1
            !For gfortran 7.2.0
            !For reasons unknown constructor function overloading is not working properly in gfortran
            !new_element => create_link(value,this%first_element,null())
            new_element => doubly_linked_list_link(value,this%first_element,null())
            call this%first_element%set_previous_element(new_element)
            this%first_element => new_element
        else
           this%length = 1
           !For gfortran 7.2.0
           !For reasons unknown constructor function overloading is not working properly in gfortran
           !this%first_element => create_link(value,null(),null())
           this%first_element => doubly_linked_list_link(value,null(),null())
           this%last_element => this%first_element
        end if
        deallocate(value)
    end subroutine add_generic_value_to_front

    subroutine add_integer_to_back(this,value)
        class(doubly_linked_list), intent(inout) :: this
        integer, intent(in) :: value
        class(*), pointer :: pointer_to_value
            allocate(pointer_to_value,source=value)
            call this%add_generic_value_to_back(pointer_to_value)
    end subroutine add_integer_to_back

    subroutine add_real_to_back(this,value)
        class(doubly_linked_list), intent(inout) :: this
        real, intent(in) :: value
        class(*), pointer :: pointer_to_value
            allocate(pointer_to_value,source=value)
            call this%add_generic_value_to_back(pointer_to_value)
    end subroutine add_real_to_back

    subroutine add_logical_to_back(this,value)
        class(doubly_linked_list), intent(inout) :: this
        logical, intent(in) :: value
        class(*), pointer :: pointer_to_value
            allocate(pointer_to_value,source=value)
            call this%add_generic_value_to_back(pointer_to_value)
    end subroutine add_logical_to_back

    subroutine add_coords_to_back(this,value)
        class(doubly_linked_list), intent(inout) :: this
        class(coords), intent(in) :: value
        class(*), pointer :: pointer_to_value
            allocate(pointer_to_value,source=value)
            call this%add_generic_value_to_back(pointer_to_value)
    end subroutine add_coords_to_back

    subroutine add_coords_to_front(this,value)
        class(doubly_linked_list), intent(inout) :: this
        class(coords), intent(in) :: value
        class(*), pointer :: pointer_to_value
            allocate(pointer_to_value,source=value)
            call this%add_generic_value_to_front(pointer_to_value)
    end subroutine add_coords_to_front

    subroutine add_integer_to_front(this,value)
        class(doubly_linked_list), intent(inout) :: this
        integer, intent(in) :: value
        class(*), pointer :: pointer_to_value
            allocate(pointer_to_value,source=value)
            call this%add_generic_value_to_front(pointer_to_value)
    end subroutine add_integer_to_front

    subroutine add_real_to_front(this,value)
        class(doubly_linked_list), intent(inout) :: this
        real, intent(in) :: value
        class(*), pointer :: pointer_to_value
            allocate(pointer_to_value,source=value)
            call this%add_generic_value_to_front(pointer_to_value)
    end subroutine add_real_to_front

    subroutine add_logical_to_front(this,value)
        class(doubly_linked_list), intent(inout) :: this
        logical,intent(in) :: value
        class(*), pointer :: pointer_to_value
            allocate(pointer_to_value,source=value)
            call this%add_generic_value_to_front(pointer_to_value)
    end subroutine add_logical_to_front

    function iterate_forward(this) result(at_end_of_list)
        class(doubly_linked_list) :: this
        class(doubly_linked_list_link), pointer :: next_element
        logical :: at_end_of_list
            at_end_of_list = .FALSE.
            if (this%length == 0) then
                at_end_of_list = .TRUE.
                return
            end if
            if  (associated(this%iterator_position)) then
                if (associated(this%iterator_position,this%last_element)) then
                    at_end_of_list = .TRUE.
                else
                    next_element => this%iterator_position%get_next_element()
                    this%iterator_position => next_element
                end if
            else
                next_element => this%first_element
                this%iterator_position => next_element
            end if
    end function iterate_forward

    function iterate_backward(this) result(at_start_of_list)
        class(doubly_linked_list) :: this
        class(doubly_linked_list_link), pointer :: previous_element
        logical :: at_start_of_list
            at_start_of_list = .FALSE.
            if  (associated(this%iterator_position)) then
                if (associated(this%iterator_position,this%first_element)) then
                    at_start_of_list = .TRUE.
                else
                    previous_element => this%iterator_position%get_previous_element()
                    this%iterator_position => previous_element
                end if
            else
                previous_element => this%last_element
                this%iterator_position => previous_element
            end if
    end function iterate_backward

    function get_value_at_iterator_position(this) result(value)
        class(doubly_linked_list), intent(in) :: this
        class(*), pointer :: value
            value => this%iterator_position%get_value()
    end function get_value_at_iterator_position

    !Intending only for testing purposes
    function get_iterator_position(this) result(position)
        class(doubly_linked_list),intent(in) :: this
        type(doubly_linked_list_link), pointer :: position
            position => this%iterator_position
    end function get_iterator_position

    function get_first_element_pointer(this) result(first_element)
        class(doubly_linked_list),intent(in) :: this
        type(doubly_linked_list_link), pointer :: first_element
            first_element => this%first_element
    end function get_first_element_pointer

    function get_last_element_pointer(this) result(last_element)
        class(doubly_linked_list),intent(in) :: this
        type(doubly_linked_list_link), pointer :: last_element
            last_element => this%last_element
    end function get_last_element_pointer

    subroutine remove_element_at_iterator_position(this)
        class(doubly_linked_list), intent(inout) :: this
        type(doubly_linked_list_link), pointer :: previous_element
        type(doubly_linked_list_link), pointer :: next_element
        if (associated(this%iterator_position)) then
            if(associated(this%iterator_position,this%first_element) .and. &
               associated(this%iterator_position,this%last_element)) then
               this%length = 0
               this%first_element => null()
               this%last_element => null()
               !In compliant compiler could use final procedure instead of this
               !line
               call this%iterator_position%destructor()
               deallocate(this%iterator_position)
               this%iterator_position => null()
            else if(associated(this%iterator_position,this%first_element)) then
                this%length = this%length - 1
                next_element     => this%iterator_position%get_next_element()
                call next_element%set_previous_element(next_element)
                this%first_element => next_element
                !In complaint compiler could use final procedure instead of this
                !line
                call this%iterator_position%destructor()
                deallocate(this%iterator_position)
                this%iterator_position => null()
            else if(associated(this%iterator_position,this%last_element)) then
                this%length = this%length - 1
                previous_element => this%iterator_position%get_previous_element()
                call previous_element%set_next_element(previous_element)
                this%last_element => previous_element
                !In complaint compiler could use final procedure instead of this
                !line
                call this%iterator_position%destructor()
                deallocate(this%iterator_position)
                this%iterator_position => previous_element
            else
                this%length = this%length - 1
                previous_element => this%iterator_position%get_previous_element()
                next_element     => this%iterator_position%get_next_element()
                call previous_element%set_next_element(next_element)
                call next_element%set_previous_element(previous_element)
                !In complaint compiler could use final procedure instead of this
                !line
                call this%iterator_position%destructor()
                deallocate(this%iterator_position)
                this%iterator_position => previous_element
            end if
        end if
    end subroutine remove_element_at_iterator_position

    subroutine reset_iterator(this)
        class(doubly_linked_list), intent(inout) :: this
            this%iterator_position=>null()
    end subroutine

end module doubly_linked_list_mod
