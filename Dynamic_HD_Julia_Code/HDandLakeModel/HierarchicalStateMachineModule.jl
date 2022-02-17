module HierarchicalStateMachineModule

using UserExceptionModule: UserError

abstract type State end

abstract type Event end

function handle_event(state::State,event::Event)
  throw(UserError())
end

mutable struct HierarchicalStateMachine
  state::State
end

function handle_event(hsm::HierarchicalStateMachine, event::Event)
  hsm.state = handle_event(hsm.state,event)
end

end
