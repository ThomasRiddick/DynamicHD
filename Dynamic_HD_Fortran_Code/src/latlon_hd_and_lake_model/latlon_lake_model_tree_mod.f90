module latlon_lake_model_tree_mod

implicit none
private

type, public :: rooted_tree
  private
  integer :: label
  type(doubly_linked_list), pointer :: direct_nodes => null()
  type(rooted_tree), pointer :: root => null()
  type(rooted_tree), pointer :: superior => null()
  contains
    procedure :: rooted_tree_destructor
    procedure :: set_root
    procedure :: set_superior
    procedure :: set_root_for_all_nodes
    procedure :: add_direct_node
    procedure :: get_direct_nodes
    procedure :: get_label
    procedure :: remove_node_from_direct_nodes
end type rooted_tree

interface rooted_tree
    module procedure rooted_tree_constructor
end interface rooted_tree

type :: doubly_linked_list_link
    private
    !> A pointer to the value occupying this place/link in the chain
    type(rooted_tree), pointer :: value => null()
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
end type doubly_linked_list_link

interface doubly_linked_list_link
    module procedure doubly_linked_list_link_constructor
end interface doubly_linked_list_link

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
        procedure :: doubly_linked_list_destructor
        !> Getter for length
        procedure :: get_length
        procedure :: add_value_to_back
        procedure :: add_value_to_front
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
    module procedure doubly_linked_list_constructor
end interface doubly_linked_list

type, public :: rooted_tree_forest
  private
  type(doubly_linked_list), pointer :: sets => null()
  contains
    procedure :: rooted_tree_forest_destructor
    procedure :: find_root_from_label
    procedure :: make_new_link_from_labels
    procedure :: split_set
    procedure :: add_set
    procedure :: contains_set
    procedure :: get_set
    procedure :: check_set_has_elements
    procedure :: get_all_node_labels_of_set
end type rooted_tree_forest

interface rooted_tree_forest
    module procedure rooted_tree_forest_constructor
end interface rooted_tree_forest

contains

  subroutine rooted_tree_destructor(this)
    class(rooted_tree) :: this
      call this%direct_nodes%doubly_linked_list_destructor()
      deallocate(this%direct_nodes)
  end subroutine rooted_tree_destructor

  subroutine set_root(this,new_root)
    class(rooted_tree) :: this
    type(rooted_tree), pointer :: new_root
      this%root => new_root
  end subroutine set_root

  subroutine set_superior(this,new_superior)
    class(rooted_tree) :: this
    type(rooted_tree), pointer :: new_superior
      this%superior => new_superior
  end subroutine set_superior

  recursive subroutine set_root_for_all_nodes(this,new_root)
    class(rooted_tree) :: this
    type(rooted_tree), pointer :: new_root
    type(rooted_tree), pointer :: i
      call this%set_root(new_root)
      call this%direct_nodes%reset_iterator()
      do while (.not. this%direct_nodes%iterate_forward())
        i => this%direct_nodes%get_value_at_iterator_position()
        call i%set_root_for_all_nodes(new_root)
      end do
  end subroutine

  subroutine add_direct_node(this,node_to_add)
    class(rooted_tree) :: this
    type(rooted_tree), pointer :: node_to_add
      call this%direct_nodes%add_value_to_back(node_to_add)
  end subroutine add_direct_node

  function get_direct_nodes(this) result(direct_nodes)
    class(rooted_tree) :: this
    type(doubly_linked_list), pointer :: direct_nodes
      direct_nodes => this%direct_nodes
  end function get_direct_nodes

  function get_label(this) result(label)
    class(rooted_tree) :: this
    integer :: label
      label = this%label
  end function get_label

  subroutine remove_node_from_direct_nodes(this,node_to_remove)
    class(rooted_tree) :: this
    type(rooted_tree), pointer, intent(in) :: node_to_remove
    type(rooted_tree), pointer :: i
      call this%direct_nodes%reset_iterator()
      do while (.not. this%direct_nodes%iterate_forward())
        i => this%direct_nodes%get_value_at_iterator_position()
        if (i%get_label() == node_to_remove%get_label()) then
          call this%direct_nodes%remove_element_at_iterator_position()
          exit
        end if
      end do
  end subroutine remove_node_from_direct_nodes

  function rooted_tree_constructor(label_in) &
      result(constructor)
      type(rooted_tree), pointer :: constructor
      integer :: label_in
          allocate(constructor)
          constructor%label = label_in
          constructor%root => constructor
          constructor%superior => constructor
          constructor%direct_nodes => doubly_linked_list()
  end function rooted_tree_constructor

 function get_value(this) result(value)
      class(doubly_linked_list_link), intent(in) :: this
      type (rooted_tree), pointer :: value
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
      type(doubly_linked_list_link), pointer :: constructor
      type(doubly_linked_list_link), pointer :: next_element
      type(doubly_linked_list_link), pointer :: previous_element
      type (rooted_tree), pointer :: value
          allocate(constructor)
          constructor%next_element => next_element
          constructor%previous_element => previous_element
          constructor%value => value
  end function doubly_linked_list_link_constructor

  subroutine doubly_linked_list_destructor(this)
      class(doubly_linked_list), intent(inout) :: this
          if (this%get_length() > 0) then
              call this%reset_iterator()
              do
                  if (this%iterate_forward()) exit
                  call this%remove_element_at_iterator_position()
              end do
          end if
  end subroutine doubly_linked_list_destructor

  pure function get_length(this) result(length)
      class(doubly_linked_list), intent(in) :: this
      integer :: length
          length = this%length
  end function get_length

  subroutine add_value_to_back(this,value)
      class(doubly_linked_list), intent(inout) :: this
      type(rooted_tree), pointer, intent(inout) :: value
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
  end subroutine add_value_to_back

  subroutine add_value_to_front(this,value)
      class(doubly_linked_list), intent(inout) :: this
      type(rooted_tree), pointer, intent(inout) :: value
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
  end subroutine add_value_to_front

  function iterate_forward(this) result(at_end_of_list)
      class(doubly_linked_list) :: this
      type(doubly_linked_list_link), pointer :: next_element
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
      type(doubly_linked_list_link), pointer :: previous_element
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
      type(rooted_tree), pointer :: value
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
             deallocate(this%iterator_position)
             this%iterator_position => null()
          else if(associated(this%iterator_position,this%first_element)) then
              this%length = this%length - 1
              next_element     => this%iterator_position%get_next_element()
              call next_element%set_previous_element(next_element)
              this%first_element => next_element
              !In complaint compiler could use final procedure instead of this
              !line
              deallocate(this%iterator_position)
              this%iterator_position => null()
          else if(associated(this%iterator_position,this%last_element)) then
              this%length = this%length - 1
              previous_element => this%iterator_position%get_previous_element()
              call previous_element%set_next_element(previous_element)
              this%last_element => previous_element
              !In complaint compiler could use final procedure instead of this
              !line
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
              deallocate(this%iterator_position)
              this%iterator_position => previous_element
          end if
      end if
  end subroutine remove_element_at_iterator_position

  subroutine reset_iterator(this)
      class(doubly_linked_list), intent(inout) :: this
          this%iterator_position=>null()
  end subroutine reset_iterator

  function doubly_linked_list_constructor() &
      result(constructor)
      type(doubly_linked_list), pointer :: constructor
          allocate(constructor)
          constructor%first_element => null()
          constructor%last_element => null()
          constructor%iterator_position => null()
          constructor%length = 0
  end function doubly_linked_list_constructor

  function find_root(target_set) result(root)
    type(rooted_tree), pointer, intent(inout) :: target_set
    type(rooted_tree), pointer :: root
    type(rooted_tree), pointer :: working_ptr
      root => target_set
      do while (root%root%label /= root%label)
        working_ptr => root%root
        call working_ptr%set_root(root%root%root)
        root => working_ptr
      end do
  end function find_root

  function find_root_from_label(this,label_in) result(root_label)
    class(rooted_tree_forest) :: this
    integer, intent(in) :: label_in
    integer :: root_label
    type(rooted_tree), pointer :: x
    type(rooted_tree), pointer :: root_x
      x => this%get_set(label_in)
      root_x => find_root(x)
      root_label = root_x%label
  end function find_root_from_label

  function make_new_link(x,y) result(success)
    type(rooted_tree), pointer, intent(inout) :: x
    type(rooted_tree), pointer, intent(inout) :: y
    type(rooted_tree), pointer :: root_x
    logical :: success
      root_x => find_root(x)
      if (y%root%label /= y%label) then
        write(*,*) "Rhs set must be tree root when adding link"
        stop
      end if
      if (root_x%label == y%label) then
        success = .False.
      else
        call y%set_root(root_x)
        call x%add_direct_node(y)
        call y%set_superior(x)
        success = .True.
      end if
  end function make_new_link

  function make_new_link_from_labels(this,label_x,label_y) result(success)
    class(rooted_tree_forest) :: this
    integer, intent(in) :: label_x
    integer, intent(in) :: label_y
    logical :: success
    type(rooted_tree), pointer :: x
    type(rooted_tree), pointer :: y
      x => this%get_set(label_x)
      y => this%get_set(label_y)
      success = make_new_link(x,y)
  end function make_new_link_from_labels

  function split_set(this,set_label,subset_to_split_label) result(success)
    class(rooted_tree_forest) :: this
    integer, intent(in) :: set_label
    integer, intent(in) :: subset_to_split_label
    logical :: success
    type(rooted_tree), pointer :: set
    type(rooted_tree), pointer :: original_root
    type(rooted_tree), pointer :: new_subset
    type(rooted_tree), pointer :: new_subset_root
    type(rooted_tree), pointer :: new_subset_superior
      set => this%get_set(set_label)
      original_root => find_root(set)
      new_subset => this%get_set(subset_to_split_label)
      new_subset_root => find_root(new_subset)
      if (new_subset_root%get_label() /= original_root%get_label()) then
        success = .False.
        return
      end if
      call new_subset%set_root_for_all_nodes(new_subset)
      new_subset_superior => new_subset%superior
      call new_subset_superior%remove_node_from_direct_nodes(new_subset)
      call new_subset%set_superior(new_subset)
      success = .True.
  end function split_set

  subroutine add_set(this,label_in)
    class(rooted_tree_forest) :: this
    integer, intent(in) :: label_in
    type(rooted_tree), pointer :: new_set
      if (.not. this%contains_set(label_in)) then
        new_set => rooted_tree(label_in)
        call this%sets%add_value_to_back(new_set)
      end if
  end subroutine add_set

  function contains_set(this,label_in)
    class(rooted_tree_forest) :: this
    integer, intent(in) :: label_in
    logical :: contains_set
    type(rooted_tree), pointer :: i
      contains_set = .False.
      call this%sets%reset_iterator()
      do while (.not. this%sets%iterate_forward())
        i => this%sets%get_value_at_iterator_position()
        if (i%get_label() == label_in) then
          contains_set = .True.
          exit
        end if
      end do
  end function contains_set

  function get_set(this,label_in) result(set)
    class(rooted_tree_forest) :: this
    integer, intent(in) :: label_in
    type(rooted_tree),pointer :: set
    type(rooted_tree), pointer :: i
      call this%sets%reset_iterator()
      do while (.not. this%sets%iterate_forward())
        i => this%sets%get_value_at_iterator_position()
        if (i%get_label() == label_in) then
          set => i
          exit
        end if
      end do
      if (.not. associated(set)) then
        write(*,*) "Requested set doesn't exist"
        stop
      end if
  end function get_set

  function check_set_has_elements(this,label_of_element, &
                                  element_labels) result(elements_match_set)
    class(rooted_tree_forest) :: this
    integer, intent(in) :: label_of_element
    integer, intent(in), dimension(:), pointer :: element_labels
    logical :: elements_match_set
    integer,  dimension(:), pointer :: node_labels
      node_labels => this%get_all_node_labels_of_set(label_of_element)
      if (size(node_labels) == size(element_labels) ) then
        elements_match_set = all(node_labels == element_labels)
      else
        elements_match_set = .False.
      end if
      deallocate(node_labels)
  end function check_set_has_elements

  function get_all_node_labels_of_set(this,label_of_element) result(label_list)
    class(rooted_tree_forest) :: this
    integer, intent(in) :: label_of_element
    integer, dimension(:), pointer :: label_list
    type(rooted_tree), pointer :: i
    type(rooted_tree), pointer :: root
    integer :: counter
      call this%sets%reset_iterator()
      counter = 0
      do while (.not. this%sets%iterate_forward())
        i => this%sets%get_value_at_iterator_position()
        root => find_root(i)
        if (root%get_label() == label_of_element) then
          counter = counter + 1
        end if
      end do
      allocate(label_list(counter))
      call this%sets%reset_iterator()
      counter = 1
      do while (.not. this%sets%iterate_forward())
        i => this%sets%get_value_at_iterator_position()
        root => find_root(i)
        if (root%get_label() == label_of_element) then
          label_list(counter) = i%get_label()
          counter = counter + 1
        end if
      end do
  end function get_all_node_labels_of_set

  subroutine rooted_tree_forest_destructor(this)
    class(rooted_tree_forest) :: this
    type(rooted_tree), pointer :: i
        call this%sets%reset_iterator()
        do while (.not. this%sets%iterate_forward())
          i => this%sets%get_value_at_iterator_position()
          if (associated(i)) then
            call i%rooted_tree_destructor()
            deallocate(i)
          end if
        end do
        call this%sets%doubly_linked_list_destructor()
        deallocate(this%sets)
  end subroutine rooted_tree_forest_destructor

  function rooted_tree_forest_constructor() &
    result(constructor)
    type(rooted_tree_forest), pointer :: constructor
      allocate(constructor)
      constructor%sets => doubly_linked_list()
  end function rooted_tree_forest_constructor

end module latlon_lake_model_tree_mod
