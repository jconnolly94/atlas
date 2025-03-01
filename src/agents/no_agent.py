# agents/no_agent.py
from .agent import Agent


class NoAgent(Agent):
    """
    A baseline agent that does not make decisions, letting SUMO's default
    traffic light controller handle everything.

    This agent acts as a control case for experiments, allowing measurement
    of performance without any RL intervention.
    """

    def __init__(self, tls_id, network):
        """Initialize the agent.

        Args:
            tls_id: ID of the traffic light this agent controls
            network: Network object providing access to simulation data
        """
        super().__init__(tls_id, network)

        # Get the default program for this traffic light
        self.default_program = 0  # Usually the first program

        # Track phase information
        try:
            self.phases = network.get_possible_phases(tls_id)
            self.current_phase = 0
        except Exception as e:
            print(f"Warning: Could not get phases for {tls_id}: {e}")
            self.phases = [0]
            self.current_phase = 0

        # For data collection purposes, we need to track current action
        self.last_action = 0

    def choose_action(self, state):
        """
        Instead of returning None, we now return a valid phase number.
        This allows it to work with the environment while still effectively
        being a no-op since we don't change the current phase.

        Args:
            state: Current state observation

        Returns:
            Current phase number as a "no-op" action
        """
        # Get current phase from TraCI if possible
        try:
            current_phase = self.network.get_current_phase(self.tls_id)
            self.current_phase = current_phase
        except:
            # If that fails, just use the first phase
            current_phase = 0 if not self.phases else self.phases[0]
            self.current_phase = current_phase

        # Return current phase as the action, effectively doing nothing
        self.last_action = current_phase
        return current_phase

    def learn(self, state, action, next_state, done):
        """
        No learning occurs in this agent.

        Args:
            state: Previous state (ignored)
            action: Action that was taken (ignored)
            next_state: Resulting state (ignored)
            done: Whether this is a terminal state (ignored)
        """
        # No learning, but we need to ensure we're tracking actions for reward calculation
        self.last_action = action if action is not None else self.last_action

    def calculate_reward(self, state, action, next_state):
        """
        Calculate reward the same way as other agents for fair comparison.

        Args:
            state: Previous state
            action: Action taken (may be None)
            next_state: Resulting state

        Returns:
            reward: Calculated reward value
            components: Dictionary of reward components
        """
        # If action is None, use the last action we tracked
        actual_action = action if action is not None else self.last_action

        # Use the standard reward calculation from the base class
        return super().calculate_reward(state, actual_action, next_state)