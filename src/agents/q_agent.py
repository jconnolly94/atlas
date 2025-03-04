# agents/q_agent.py
from .agent import Agent
import random
from collections import defaultdict
from typing import Dict, Any


class QAgent(Agent):
    """Q-learning agent for traffic signal control."""
    
    # Default configuration
    DEFAULT_CONFIG = {
        "alpha": 0.2,             # Learning rate
        "gamma": 0.9,             # Discount factor
        "epsilon": 0.7,           # Significantly increased for much more exploration
        "state_bin_size": 5
    }

    def __init__(self, tls_id, network, alpha=0.1, gamma=0.9, epsilon=0.1, state_bin_size=5):
        """Initialize the agent.

        Args:
            tls_id: ID of the traffic light this agent controls
            network: Network object providing access to simulation data
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            state_bin_size: Size of state bins for discretization
        """
        super().__init__(tls_id, network)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.state_bin_size = state_bin_size

        # Get possible actions for this traffic light
        self.possible_actions = network.get_possible_phases(tls_id)

        # Q-value table
        self.q_values = defaultdict(float)

        # Track last action for reward calculation
        self.last_action = None
        
    @classmethod
    def create(cls, tls_id, network, **kwargs):
        """Create an instance of the QAgent with proper configuration.
        
        Args:
            tls_id: ID of the traffic light this agent controls
            network: Network object providing access to simulation data
            **kwargs: Additional configuration parameters
            
        Returns:
            Properly configured QAgent instance
        """
        # Start with default configuration
        config = cls.DEFAULT_CONFIG.copy()
        
        # Override with provided kwargs
        config.update(kwargs)
        
        # Create and return instance
        return cls(tls_id, network, **config)

    def _discretize_state(self, state):
        """Discretize the state for Q-table lookup.

        The state is a vector of continuous values, but Q-learning
        requires a discrete state space. This method converts the
        continuous state to a discrete representation.

        Args:
            state: Continuous state vector

        Returns:
            Discretized state representation (int or tuple)
        """
        # Use only the first feature (waiting_time) for simplicity
        # Could be extended to use more features
        return int(state[0] // self.state_bin_size)

    def choose_action(self, state):
        """Choose an action based on the current state.

        Uses epsilon-greedy policy: with probability epsilon, choose
        a random action; otherwise, choose the action with the highest
        Q-value for the current state.

        Args:
            state: Current state observation

        Returns:
            Selected action (phase index)
        """
        # Discretize the state
        discrete_state = self._discretize_state(state)

        # Exploration: choose random action with probability epsilon
        if random.random() < self.epsilon:
            return random.choice(self.possible_actions)

        # Exploitation: choose best action according to Q-values
        best_q = float('-inf')
        best_action = None

        for action in self.possible_actions:
            q_val = self.q_values[(discrete_state, action)]
            if q_val > best_q:
                best_q = q_val
                best_action = action

        # If no action has been learned yet, choose randomly
        if best_action is None:
            return random.choice(self.possible_actions)

        return best_action

    def learn(self, state, action, next_state, done):
        """Learn from experience using Q-learning.

        Updates Q-values based on the observed transition and reward.

        Args:
            state: Previous state
            action: Action that was taken
            next_state: Resulting state
            done: Whether this is a terminal state
        """
        # Calculate reward
        reward, _ = self.calculate_reward(state, action, next_state)

        # Discretize states
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)

        # Current Q-value
        old_q = self.q_values[(discrete_state, action)]

        # Maximum Q-value for next state
        max_q_new = max(
            (self.q_values.get((discrete_next_state, a), 0)
             for a in self.possible_actions),
            default=0
        )

        # Q-learning update rule
        new_q = old_q + self.alpha * (reward + self.gamma * max_q_new - old_q)

        # Update Q-value
        self.q_values[(discrete_state, action)] = new_q

    def calculate_reward(self, state, action, next_state):
        """Calculate the reward for a transition.

        This implementation uses a simple reward based on waiting time reduction.

        Args:
            state: Previous state
            action: Action that was taken
            next_state: Resulting state

        Returns:
            reward: Calculated reward value
            components: Dictionary of reward components
        """
        # Use parent's default reward calculation
        return super().calculate_reward(state, action, next_state)