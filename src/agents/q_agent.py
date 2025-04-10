# agents/q_agent.py
from .agent import Agent
import random
import numpy as np
from collections import defaultdict
from typing import Dict, Any, Tuple, List, Optional, Union
# Add necessary imports for state persistence
import os
import pickle
import json
import logging

# Configure logger
q_agent_logger = logging.getLogger(__name__)


class QAgent(Agent):
    """Q-learning agent for traffic signal control using phase-based control."""
    
    # Default configuration updated with learnings from DQN agent
    DEFAULT_CONFIG = {
        "alpha": 0.1,             # Learning rate (adjusted for Q-learning)
        "gamma": 0.95,            # Discount factor (from DQN)
        "epsilon": 1.0,           # Start with full exploration (from DQN)
        "epsilon_decay": 0.999,   # Slower decay for better exploration (from DQN)
        "min_epsilon": 0.05       # Lower minimum exploitation (from DQN)
    }

    def __init__(self, tls_id, network, alpha=0.1, gamma=0.95, epsilon=1.0, 
                 epsilon_decay=0.999, min_epsilon=0.05):
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
        self.last_action = None  # Will store the chosen phase index
        self.current_phase_start_time = 0.0  # Track time for state features
        
        # --- Action Space: Phases ---
        try:
            # Get phase definitions from network interface
            self.phases = self.network.get_traffic_light_phases(self.tls_id)
            if not self.phases:
                raise ValueError(f"No program logics found for TLS '{self.tls_id}'")
            self.num_phases = len(self.phases)
            self.action_size = self.num_phases  # Action is selecting a phase index
            q_agent_logger.info(f"TLS {self.tls_id} initialized with {self.num_phases} phases.")
        except Exception as e:
            q_agent_logger.error(f"Failed to get phases for TLS {self.tls_id}: {e}. Setting num_phases to 4 as fallback.")
            # Fallback if network interface fails
            self.num_phases = 4  # Common fallback
            self.action_size = 4
            self.phases = None  # Indicate phases couldn't be loaded
        
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

    def choose_action(self, state: Dict[str, Any]) -> Optional[int]:
        """
        Choose a phase index based on the current state.
        
        Args:
            state: Current state observation with link_states and current phase
            
        Returns:
            Phase index to set, or None if no action needed
        """
        # Ensure state has the required format
        if not isinstance(state, dict) or 'link_states' not in state:
            q_agent_logger.warning(f"Agent received incompatible state format. Expected dict with 'link_states' key.")
            return None
            
        # Create a discretized state representation for the Q-table
        state_key = self._discretize_state(state)
        
        # Get current phase index
        try:
            current_phase_index = state.get('current_phase_index', 
                                          self.network.get_current_phase_index(self.tls_id))
        except Exception as e:
            q_agent_logger.warning(f"Could not get current phase index: {e}")
            current_phase_index = 0  # Default if not available
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Exploration: choose a random phase index
            action_index = random.randrange(self.action_size)
            
            # Optionally: avoid selecting the current phase during exploration
            # to enhance exploration effectiveness
            if action_index == current_phase_index and self.action_size > 1:
                other_phases = [i for i in range(self.action_size) if i != current_phase_index]
                action_index = random.choice(other_phases)
        else:
            # Exploitation: choose the best phase based on Q-values
            if state_key not in self.q_values or not self.q_values[state_key]:
                # If state not seen before or no value found
                # Choose a phase based on queue lengths
                link_states = state.get('link_states', [])
                
                if not link_states:
                    # No link states available, choose random phase
                    action_index = random.randrange(self.action_size)
                else:
                    # Heuristic: simply move to next phase when no data available
                    action_index = (current_phase_index + 1) % self.action_size
            else:
                # Find the action with highest Q-value
                action_index = int(max(self.q_values[state_key], 
                                     key=lambda k: self.q_values[state_key][k]))
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Store the chosen phase index and update timing
        self.last_action = action_index
        current_time = self.network.get_current_time()
        if action_index != current_phase_index:
            self.current_phase_start_time = current_time

        return action_index

    def _discretize_state(self, state: Dict[str, Any]) -> Tuple:
        """
        Discretize the state for Q-table lookup, using phase-based representation.
        
        Args:
            state: State with link_states and current_phase_index
            
        Returns:
            Discretized state tuple that can be used as a dictionary key
        """
        if not isinstance(state, dict) or 'link_states' not in state:
            q_agent_logger.warning(f"Invalid state format in _discretize_state")
            return (0, (0,))  # Default state representation
            
        # Extract current phase index and link states
        current_phase_index = state.get('current_phase_index', 0)
        link_states = state.get('link_states', [])
        
        # Discretize queue lengths into bins
        queue_bins = []
        for link in sorted(link_states, key=lambda x: x.get('index', 0)):
            queue = link.get('queue_length', 0)
            # More granular queue binning (from 0-3 bins to 0-4 bins)
            if queue == 0:
                bin_value = 0
            elif queue <= 2:
                bin_value = 1
            elif queue <= 5:
                bin_value = 2
            elif queue <= 10:
                bin_value = 3
            else:
                bin_value = 4
            queue_bins.append(bin_value)
        
        # Create a tuple representation that can be used as dictionary key
        state_tuple = (
            current_phase_index,  # Current phase index (instead of signal string)
            tuple(queue_bins)     # Discretized queue lengths
        )
        
        return state_tuple

    def learn(self, state, action_index, next_state, done):
        """
        Learn from experience using Q-learning with phase-based actions.
        
        Args:
            state: Previous state
            action_index: Phase index that was selected
            next_state: Resulting state
            done: Whether this is a terminal state
        """
        # Skip learning if no action was taken
        if action_index is None:
            return
            
        # Calculate reward using the new phase-based method
        reward, _ = self.calculate_reward_phase_based(state, action_index, next_state)
        
        # Discretize states
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)
        
        # Convert action_index to string for dictionary key
        action_key = str(action_index)
        
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

    def calculate_reward_phase_based(self, state, action_index, next_state):
        """
        Calculate reward based on state changes for phase-based control.
        Adapted from the DQN agent's successful reward function.
        
        Args:
            state: Previous state
            action_index: Phase index that was selected
            next_state: Next state
            
        Returns:
            reward: Calculated reward value
            components: Dictionary of reward components
        """
        if not isinstance(state, dict) or 'link_states' not in state or \
           not isinstance(next_state, dict) or 'link_states' not in next_state:
            return 0.0, {}

        prev_link_states = state.get('link_states', [])
        next_link_states = next_state.get('link_states', [])

        if not prev_link_states or not next_link_states:
            return 0.0, {}

        # Calculate waiting time reduction (primary reward signal)
        prev_total_wait = sum(link.get('waiting_time', 0) for link in prev_link_states)
        next_total_wait = sum(link.get('waiting_time', 0) for link in next_link_states)
        wait_reward = (prev_total_wait - next_total_wait) / 10.0  # Scale down reward

        # Calculate intersection-specific throughput
        prev_vehicle_count = sum(link.get('vehicle_count', 0) for link in prev_link_states)
        next_vehicle_count = sum(link.get('vehicle_count', 0) for link in next_link_states)
        
        # Count vehicles that have passed through this intersection
        # If vehicle count decreases, it means vehicles have exited the links controlled by this TLS
        passed_vehicles = max(0, prev_vehicle_count - next_vehicle_count)
        
        # If we have link exit counts directly, use those instead
        prev_exit_count = sum(link.get('exit_count', 0) for link in prev_link_states)
        next_exit_count = sum(link.get('exit_count', 0) for link in next_link_states)
        if next_exit_count > prev_exit_count:
            passed_vehicles = next_exit_count - prev_exit_count
        
        # Reward for intersection throughput (important for continued learning)
        throughput_reward = passed_vehicles * 0.2  # Scale for appropriate impact
        
        # Small penalty for very long queues to discourage gridlock
        max_queue = 0
        for link in next_link_states:
            max_queue = max(max_queue, link.get('queue_length', 0))

        queue_penalty = 0
        if max_queue > 15:  # Penalize if any queue gets very long
            queue_penalty = -(max_queue - 15) * 0.05

        # Combine rewards
        total_reward = wait_reward + throughput_reward + queue_penalty

        components = {
            'wait_time_reduction': wait_reward,
            'throughput_reward': throughput_reward,
            'queue_penalty': queue_penalty,
            'passed_vehicles': passed_vehicles
        }

        return total_reward, components
        
    # Keep the old reward function as a fallback
    def calculate_reward(self, state, action, next_state):
        """Legacy reward calculation for backward compatibility."""
        # For link-level actions, the action might be a tuple
        if isinstance(action, tuple) and len(action) == 2:
            # Original link-level calculation
            link_index, new_state = action
            # Get metrics for all links
            next_link_states = next_state.get('link_states', [])
            total_waiting_time = sum(link.get('waiting_time', 0) for link in next_link_states)
            # Simple reward: negative waiting time
            return -total_waiting_time / 100.0, {'waiting_time': -total_waiting_time}
        else:
            # If it's a phase-based action (int), use the new reward function
            return self.calculate_reward_phase_based(state, action, next_state)

    def save_state(self, directory_path: str):
        """Saves the Q-agent's state (Q-table and hyperparameters) to the specified directory."""
        try:
            super().save_state(directory_path) # Ensure directory exists via base class call if it does that
        except AttributeError: # If Agent base class doesn't have save_state (it should)
             os.makedirs(directory_path, exist_ok=True) # Manually ensure dir exists

        q_table_path = os.path.join(directory_path, 'q_table.pkl')
        hyperparams_path = os.path.join(directory_path, 'hyperparams.json')
        q_agent_logger.info(f"Attempting to save QAgent state to {directory_path}")

        try:
            # Save Q-table (convert defaultdict to dict for potentially better compatibility)
            # Important: Ensure self.q_values exists and is populated correctly before saving
            if hasattr(self, 'q_values') and self.q_values:
                 q_values_dict = dict(self.q_values)
                 # Further convert inner defaultdicts if they exist
                 for k, inner_dd in q_values_dict.items():
                     if isinstance(inner_dd, defaultdict):
                         q_values_dict[k] = dict(inner_dd)

                 with open(q_table_path, 'wb') as f:
                    pickle.dump(q_values_dict, f)
            else:
                 q_agent_logger.warning("Q-table ('q_values') is empty or missing, skipping save.")
                 # Optionally save an empty file or handle differently if needed


            # Save hyperparameters
            hyperparams = {
                'epsilon': getattr(self, 'epsilon', None), # Use getattr for safety
                'alpha': getattr(self, 'alpha', None),
                'gamma': getattr(self, 'gamma', None),
                'epsilon_decay': getattr(self, 'epsilon_decay', None),
                'min_epsilon': getattr(self, 'min_epsilon', None),
                'num_phases': getattr(self, 'num_phases', None),
                'action_size': getattr(self, 'action_size', None)
                # Add any other relevant state variables specific to QAgent
            }
            with open(hyperparams_path, 'w') as f:
                json.dump(hyperparams, f, indent=4)
            q_agent_logger.info(f"QAgent state saved successfully to {directory_path}")

        except Exception as e:
            q_agent_logger.error(f"Error saving QAgent state to {directory_path}: {e}", exc_info=True) # Log traceback


    def load_state(self, directory_path: str):
        """Loads the Q-agent's state (Q-table and hyperparameters) from the specified directory."""
        q_table_path = os.path.join(directory_path, 'q_table.pkl')
        hyperparams_path = os.path.join(directory_path, 'hyperparams.json')
        q_agent_logger.info(f"Attempting to load QAgent state from {directory_path}")

        # Check if required files exist
        if not os.path.exists(q_table_path):
            q_agent_logger.warning(f"Cannot load QAgent state: Q-table file not found at {q_table_path}")
            return # Don't attempt partial load if essential Q-table is missing
        if not os.path.exists(hyperparams_path):
             q_agent_logger.warning(f"Cannot load QAgent state: Hyperparameters file not found at {hyperparams_path}")
             # Decide if loading only Q-table is acceptable or return here too.
             # For consistency, let's return if hyperparams are also missing.
             return

        try:
            # Load Q-table
            with open(q_table_path, 'rb') as f:
                loaded_dict = pickle.load(f)
                # Restore to nested defaultdict structure
                # Assumes the structure is state_key -> action_key -> q_value (float)
                self.q_values = defaultdict(lambda: defaultdict(float))
                for state_key, action_dict in loaded_dict.items():
                    inner_dd = defaultdict(float)
                    if isinstance(action_dict, dict): # Check if inner part is a dict
                        inner_dd.update(action_dict)
                    self.q_values[state_key] = inner_dd


            # Load hyperparameters
            with open(hyperparams_path, 'r') as f:
                hyperparams = json.load(f)
                # Use loaded values, falling back to existing values if key is missing in JSON
                self.epsilon = hyperparams.get('epsilon', getattr(self, 'epsilon', 0.1)) # Provide default for getattr too
                self.alpha = hyperparams.get('alpha', getattr(self, 'alpha', 0.1))
                self.gamma = hyperparams.get('gamma', getattr(self, 'gamma', 0.95))
                self.epsilon_decay = hyperparams.get('epsilon_decay', getattr(self, 'epsilon_decay', 0.999))
                self.min_epsilon = hyperparams.get('min_epsilon', getattr(self, 'min_epsilon', 0.05))
                
                # If num_phases is different from what's loaded, we might have compatibility issues
                loaded_num_phases = hyperparams.get('num_phases')
                loaded_action_size = hyperparams.get('action_size')
                
                if loaded_num_phases and loaded_num_phases != self.num_phases:
                    q_agent_logger.warning(f"Loaded model has {loaded_num_phases} phases, "
                                         f"but current environment has {self.num_phases}. "
                                         f"This may affect performance.")

            q_agent_logger.info(f"QAgent state loaded successfully from {directory_path}")

        except FileNotFoundError:
             # This case should ideally be caught by the checks above, but handle defensively
             q_agent_logger.error(f"Error loading QAgent state: File not found during load attempt in {directory_path}")
        except Exception as e:
            q_agent_logger.error(f"Error loading QAgent state from {directory_path}: {e}", exc_info=True) # Log traceback
            # Consider re-initializing q_values to a default state if loading fails critically
            # self.q_values = defaultdict(lambda: defaultdict(float))