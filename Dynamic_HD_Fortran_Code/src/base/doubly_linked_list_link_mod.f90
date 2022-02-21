module doubly_linked_list_link_mod
implicit none
private
!For gfortran 7.2.0
!Necessary to make constructor public as overloading default constructor is not
!working reliably in gfortran7.2.0
!public :: doubly_linked_list_link, doubly_linked_list_link_constructor
public :: doubly_linked_list_link

!> Class that implements links for a double linked list (a list that can be
!! iterated through in either direction) for any variable type
type :: doubly_linked_list_link
    private
    !> A pointer to the value occupying this place/link in the chain
    class (*), pointer :: value => null()
    !> A pointer to the previous link object in the chain
    type(doubly_linked_list_link), pointer :: previous_element => null()
    !> A pointer to the next link object in the chain
    type(doubly_linked_list_link), pointer :: next_element => null()
    contains
        !> Returns (a pointer to) the value of this link
        procedure :: get_value
        !> Returns a pointer to the next link
        procedure :: get_next_element
        !> Returns a pointer to the previous link
        procedure :: get_previous_element
        !> Set the pointer to the next link
        procedure :: set_next_element
        !> Set the pointer to the previous link
        procedure :: set_previous_element
        ! In lieu of a final routine as this feature is not currently (August 2016)
        ! supported by all fortran compilers
        ! final :: destructor
        !> Class destructor. Deallocates the value occupying this link
        procedure :: destructor
end type doubly_linked_list_link

interface doubly_linked_list_link
    module procedure doubly_linked_list_link_constructor
end interface doubly_linked_list_link

contains

    function get_value(this) result(value)
        class(doubly_linked_list_link), intent(in) :: this
        class (*), pointer :: value
            value => this%value
    end function get_value

    function get_next_element(this) result (next_element)
        class(doubly_linked_list_link), intent(in) :: this
        type(doubly_linked_list_link), pointer :: next_element
        next_element => this%next_element
    end function get_next_element

    function get_previous_element(this) result (previous_element)
        class(doubly_linked_list_link), intent(in) :: this
        type(doubly_linked_list_link), pointer :: previous_element
        previous_element => this%previous_element
    end function get_previous_element

    subroutine set_next_element(this,next_element)
        class(doubly_linked_list_link) :: this
        type(doubly_linked_list_link), pointer :: next_element
            this%next_element => next_element
    end subroutine set_next_element

    subroutine set_previous_element(this,previous_element)
        class(doubly_linked_list_link) :: this
        type(doubly_linked_list_link), pointer :: previous_element
            this%previous_element => previous_element
    end subroutine set_previous_element

    function doubly_linked_list_link_constructor(value,next_element,previous_element) &
        result(constructor)
        class(doubly_linked_list_link), pointer :: constructor
        type(doubly_linked_list_link), pointer :: next_element
        type(doubly_linked_list_link), pointer :: previous_element
        class (*), pointer :: value
            allocate(constructor)
            constructor%next_element => next_element
            constructor%previous_element => previous_element
            allocate(constructor%value,source=value)
    end function doubly_linked_list_link_constructor

    subroutine destructor(this)
        class(doubly_linked_list_link), intent(inout) :: this
        if(associated(this%value)) then
            deallocate(this%value)
        end if
    end subroutine destructor

end module doubly_linked_list_link_mod
