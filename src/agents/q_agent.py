# agents/q_agent.py
from .agent import Agent
import random
import numpy as np
from collections import defaultdict
from typing import Dict, Any, Tuple, List, Optional, Union


class QAgent(Agent):
    """Q-learning agent for traffic signal control using link-level control."""
    
    # Default configuration
    DEFAULT_CONFIG = {
        "alpha": 0.2,             # Learning rate
        "gamma": 0.9,             # Discount factor
        "epsilon": 0.7,           # High exploration rate
        "epsilon_decay": 0.995,   # Epsilon decay rate
        "min_epsilon": 0.1        # Minimum exploration rate
    }

    def __init__(self, tls_id, network, alpha=0.2, gamma=0.9, epsilon=0.7, 
                 epsilon_decay=0.995, min_epsilon=0.1):
        """Initialize the agent.

        Args:
            tls_id: ID of the traffic light this agent controls
            network: Network object providing access to simulation data
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_decay: Rate at which epsilon decreases
            min_epsilon: Minimum value for epsilon
        """
        super().__init__(tls_id, network)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Q-value dictionary (with defaultdict to avoid key errors)
        self.q_values = defaultdict(lambda: defaultdict(float))

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

    def choose_action(self, state: Dict[str, Any]) -> Optional[Tuple[int, str]]:
        """Choose a link and state change based on the current state.
        
        Args:
            state: Current state observation with link_states
            
        Returns:
            Tuple of (link_index, new_state) or None if no action needed
        """
        # Ensure state has the required format
        if not isinstance(state, dict) or 'link_states' not in state:
            print(f"Warning: Agent received incompatible state format. Expected dict with 'link_states' key.")
            return None
            
        # Extract link states and current signal state
        link_states = state['link_states']
        current_signal_state = state['current_signal_state']
        
        # No action if no links to control
        if not link_states:
            return None
        
        # Create a discretized state representation for the Q-table
        state_key = self._discretize_state(state)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Exploration: High epsilon promotes extensive exploration
            
            # First step: Choose a link to modify
            # Weighted selection based on queue length for more meaningful exploration
            queue_lengths = [max(0.1, link['queue_length']) for link in link_states]
            total_queue = sum(queue_lengths)
            
            if total_queue > 0:
                # Normalize to create probability distribution
                probs = [q/total_queue for q in queue_lengths]
                selected_link = random.choices(link_states, weights=probs, k=1)[0]
            else:
                # Equal probability if no queues
                selected_link = random.choice(link_states)
            
            link_index = selected_link['index']
            current_link_state = current_signal_state[link_index]
            
            # Second step: Choose a new state different from the current one
            if current_link_state in 'Gg':
                # If green, try red (requires yellow transition handled by network)
                new_state = 'r'
            else:
                # If not green, try green (with high probability) or keep current
                new_state = random.choices(['G', current_link_state], weights=[0.8, 0.2], k=1)[0]
            
            action = (link_index, new_state)
        else:
            # Exploitation: choose the best action for this state
            if state_key not in self.q_values or not self.q_values[state_key]:
                # If state not seen before or no value found, fallback to simple heuristic
                
                # Choose the link with the longest queue
                max_queue_link = max(link_states, key=lambda x: x['queue_length'])
                link_index = max_queue_link['index']
                current_link_state = current_signal_state[link_index]
                
                # Set to green if not already green and has queue
                if current_link_state not in 'Gg' and max_queue_link['queue_length'] > 0:
                    new_state = 'G'
                else:
                    # Switch to red if green with no queue
                    new_state = 'r'
                    
                action = (link_index, new_state)
            else:
                # Find the action with highest Q-value
                best_action_key = max(self.q_values[state_key], key=self.q_values[state_key].get)
                # Convert from string representation back to tuple
                action = eval(best_action_key)  # Safe since action_key follows our own format
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        self.last_action = action
        return action

    def _discretize_state(self, state: Dict[str, Any]) -> Tuple:
        """Discretize the state for Q-table lookup.
        
        Args:
            state: State with link_states
            
        Returns:
            Discretized state tuple that can be used as a dictionary key
        """
        # Extract current signal state and link metrics
        current_signal_state = state['current_signal_state']
        link_states = state['link_states']
        
        # Discretize queue lengths into bins
        queue_bins = []
        for link in sorted(link_states, key=lambda x: x['index']):
            queue = link['queue_length']
            if queue == 0:
                bin_value = 0
            elif queue <= 3:
                bin_value = 1
            elif queue <= 8:
                bin_value = 2
            else:
                bin_value = 3
            queue_bins.append(bin_value)
        
        # Create a tuple representation that can be used as dictionary key
        state_tuple = (
            current_signal_state,  # Current signal configuration
            tuple(queue_bins)      # Discretized queue lengths
        )
        
        return state_tuple

    def learn(self, state, action, next_state, done):
        """Learn from experience using Q-learning.
        
        Updates Q-values based on the observed transition and reward.
        
        Args:
            state: Previous state
            action: Action that was taken
            next_state: Resulting state
            done: Whether this is a terminal state
        """
        # Skip learning if no action was taken
        if action is None:
            return
            
        # Calculate reward
        reward, _ = self.calculate_reward(state, action, next_state)
        
        # Discretize states
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)
        
        # Convert action to string for dictionary key
        action_key = str(action)
        
        # Current Q-value
        old_q = self.q_values[state_key][action_key]
        
        # Find max Q-value for next state
        next_q_max = 0.0
        if next_state_key in self.q_values and self.q_values[next_state_key]:
            next_q_max = max(self.q_values[next_state_key].values())
        
        # Q-learning update rule
        new_q = old_q + self.alpha * (reward + self.gamma * next_q_max - old_q)
        
        # Update Q-value
        self.q_values[state_key][action_key] = new_q

    def calculate_reward(self, state, action, next_state):
        """Calculate reward for taking a lane-level action.
        
        Focuses purely on outcomes without encoding assumptions about
        "good" signal patterns.
        
        Args:
            state: Previous state
            action: Tuple of (link_index, new_state)
            next_state: Next state
            
        Returns:
            reward: Calculated reward value
            components: Dictionary of reward components
        """
        if action is None:
            return 0.0, {}  # No action, no reward
            
        link_index, new_state = action
        
        # Get metrics for all links
        next_link_states = next_state['link_states']
        
        # Find the specific link that was changed
        target_link = None
        for link in next_link_states:
            if link['index'] == link_index:
                target_link = link
                break
                
        if not target_link:
            return 0.0, {}  # Link not found
        
        # Metrics for the targeted link
        target_waiting_time = target_link['waiting_time']
        target_queue_length = target_link['queue_length']
        
        # Overall metrics across all links
        total_waiting_time = sum(link['waiting_time'] for link in next_link_states)
        max_queue_length = max((link['queue_length'] for link in next_link_states), default=0)
        total_throughput = self.network.get_departed_vehicles_count()
        
        # Normalization constants
        MAX_WAITING_TIME = 500.0
        MAX_QUEUE_LENGTH = 20.0
        MAX_THROUGHPUT = 25.0
        
        # Normalize metrics
        norm_target_waiting = min(1.0, target_waiting_time / MAX_WAITING_TIME)
        norm_total_waiting = min(1.0, total_waiting_time / (MAX_WAITING_TIME * len(next_link_states)))
        norm_max_queue = min(1.0, max_queue_length / MAX_QUEUE_LENGTH)
        norm_throughput = min(1.0, total_throughput / MAX_THROUGHPUT)
        
        # Component weights
        W_TARGET_WAITING = 0.3    # Waiting time for target link
        W_TOTAL_WAITING = 0.3     # Total waiting time across all links
        W_THROUGHPUT = 0.3        # Overall throughput 
        W_MAX_QUEUE = 0.1         # Maximum queue length (prevent extremes)
        
        # Calculate components (negative waiting times = penalties)
        target_waiting_component = -norm_target_waiting * W_TARGET_WAITING
        total_waiting_component = -norm_total_waiting * W_TOTAL_WAITING
        throughput_component = norm_throughput * W_THROUGHPUT
        max_queue_component = -norm_max_queue * W_MAX_QUEUE
        
        # Combine components
        total_reward = (
            target_waiting_component +
            total_waiting_component +
            throughput_component +
            max_queue_component
        )
        
        # Return reward and components for analysis
        components = {
            'target_waiting_component': target_waiting_component,
            'total_waiting_component': total_waiting_component,
            'throughput_component': throughput_component,
            'max_queue_component': max_queue_component
        }
        
        return total_reward, components