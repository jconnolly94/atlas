# agents/no_agent.py
from .agent import Agent
from typing import Dict, Any, Tuple, Optional
import random
import os  # Add import for file operations


class NoAgent(Agent):
    """
    A baseline agent that does not make decisions, letting SUMO's default
    traffic light controller handle everything.

    This agent acts as a control case for experiments, allowing measurement
    of performance without any RL intervention.
    
    Adapted for lane-level control compatibility while still functioning
    as a passive observer rather than an active controller.
    """
    
    # Default configuration (empty, since NoAgent doesn't need parameters)
    DEFAULT_CONFIG = {}

    def __init__(self, tls_id, network):
        """Initialize the agent.

        Args:
            tls_id: ID of the traffic light this agent controls
            network: Network object providing access to simulation data
        """
        super().__init__(tls_id, network)

        # For lane-level control compatibility
        self.last_action = None
        self.tls_id = tls_id
        self.network = network
        
    @classmethod
    def create(cls, tls_id, network, **kwargs):
        """Create an instance of the NoAgent.
        
        Args:
            tls_id: ID of the traffic light this agent controls
            network: Network object providing access to simulation data
            **kwargs: Additional configuration parameters (ignored)
            
        Returns:
            NoAgent instance
        """
        # For NoAgent, we can ignore any configuration parameters
        return cls(tls_id, network)

    def choose_action(self, state: Dict[str, Any]) -> Optional[Tuple[int, str]]:
        """
        Returns None to indicate no action, effectively letting the default
        controller handle everything. For compatibility with lane-level control,
        it still accepts and properly handles the new state format.

        Args:
            state: Current state observation in lane-level format

        Returns:
            None to indicate no action should be taken
        """
        # Validate state format for lane-level control
        if not isinstance(state, dict) or 'link_states' not in state:
            print(f"Warning: NoAgent received incompatible state format. Expected dict with 'link_states' key.")
            return None
            
        # No action is returned, indicating we're letting SUMO handle traffic lights
        return None

    def learn(self, state, action, next_state, done):
        """
        No learning occurs in this agent.

        Args:
            state: Previous state (ignored)
            action: Action that was taken (ignored)
            next_state: Resulting state (ignored)
            done: Whether this is a terminal state (ignored)
        """
        # No learning needed for baseline agent
        pass

    def calculate_reward(self, state, action, next_state):
        """
        Calculate a simple reward to track baseline performance metrics.
        This doesn't use agent-specific reward functions and is purely for 
        measurement and comparison purposes.

        Args:
            state: Previous state in lane-level format
            action: Action taken (always None for NoAgent)
            next_state: Resulting state in lane-level format

        Returns:
            reward: Simple performance metric (-total_waiting_time)
            components: Dictionary with raw metrics (no reward formula applied)
        """
        # For lane-level state format
        if isinstance(next_state, dict) and 'link_states' in next_state:
            # Simply capture raw metrics without applying a reward formula
            next_link_states = next_state['link_states']
            
            # Collect key metrics
            total_waiting_time = sum(link['waiting_time'] for link in next_link_states)
            total_queue_length = sum(link['queue_length'] for link in next_link_states)
            max_queue_length = max((link['queue_length'] for link in next_link_states), default=0)
            
            # Get throughput if network is available
            throughput = self.network.get_departed_vehicles_count() if self.network else 0
                
            # Return raw metrics in components dictionary for data collection and analysis
            components = {
                'total_waiting_time': total_waiting_time,
                'total_queue_length': total_queue_length,
                'max_queue_length': max_queue_length,
                'throughput': throughput
            }
            
            # For reward, use a simple metric that doesn't bias toward any specific agent's approach
            # This is just for numerical comparison and doesn't affect agent behavior
            reward = -total_waiting_time / 100.0  # Simple scaling to keep values reasonable
            
            return reward, components
        
        # For legacy state format (fall back to very simple calculation)
        elif isinstance(next_state, (list, tuple)) and len(next_state) >= 4:
            waiting_time = next_state[0]
            throughput = next_state[3]
            
            components = {
                'waiting_time': waiting_time,
                'throughput': throughput
            }
            
            reward = -waiting_time / 100.0  # Simple scaling to keep values reasonable
            
            return reward, components
            
        # If neither format works, return zero
        return 0.0, {}
        
    def save_state(self, directory_path: str):
        """
        No state to save for NoAgent. Ensures directory exists as per base class contract.
        """
        # Call base method to ensure directory exists if it does that, or create manually
        try:
            super().save_state(directory_path)
        except AttributeError:  # Fallback if Agent base class save_state only has pass
            os.makedirs(directory_path, exist_ok=True)
        # No actual state saving needed
        pass

    def load_state(self, directory_path: str):
        """
        No state to load for NoAgent. Does nothing.
        """
        # No state loading needed
        pass