from abc import ABC, abstractmethod
from typing import Dict, Any, ClassVar, List, Tuple, Optional


class Agent(ABC):
    """Abstract base class for all agents.

    Each agent implementation should handle its own:
    1. State processing
    2. Action selection
    3. Learning mechanism
    4. Reward calculation
    
    Supports both legacy phase-based control and modern lane-level control.
    """

    # Default configurations for agent subclasses
    DEFAULT_CONFIG: ClassVar[Dict[str, Any]] = {}

    def __init__(self, tls_id, network):
        """Initialize the agent.

        Args:
            tls_id: ID of the traffic light this agent controls
            network: Network object providing access to simulation data
        """
        self.tls_id = tls_id
        self.network = network
        self.last_action = None

    @classmethod
    def create(cls, tls_id, network, **kwargs):
        """Create an instance of the agent with proper configuration.
        
        This class method allows each agent subclass to handle its own
        initialization logic, including default parameters and any
        special configuration needs.
        
        Args:
            tls_id: ID of the traffic light this agent controls
            network: Network object providing access to simulation data
            **kwargs: Additional configuration parameters
            
        Returns:
            Properly configured agent instance
        """
        # Start with default configuration
        config = cls.DEFAULT_CONFIG.copy()
        
        # Override with provided kwargs
        config.update(kwargs)
        
        # Create and return instance
        return cls(tls_id, network, **config)
        
    @abstractmethod
    def choose_action(self, state: Dict[str, Any]) -> Optional[Tuple[int, str]]:
        """Choose an action based on the current state.

        For lane-level control, should return a tuple of (link_index, new_state)
        where link_index identifies the specific link to change, and new_state
        is the signal state to apply (e.g., 'G', 'r', etc.).

        Args:
            state: Current state observation (dictionary with link_states and current_signal_state)

        Returns:
            Tuple of (link_index, new_state) or None if no action
        """
        pass

    @abstractmethod
    def learn(self, state: Dict[str, Any], action: Optional[Tuple[int, str]], 
              next_state: Dict[str, Any], done: bool) -> None:
        """Learn from experience.

        Args:
            state: Previous state dictionary
            action: Action that was taken as (link_index, new_state)
            next_state: Resulting state dictionary
            done: Whether this is a terminal state
        """
        pass
        
    def get_adjacent_traffic_lights(self, network) -> List[str]:
        """Determine adjacent traffic lights based on network topology.
        
        This method delegates to the network's implementation of adjacent
        traffic light determination.
        
        Args:
            network: Network object providing access to simulation data
            
        Returns:
            List of adjacent traffic light IDs
        """
        return network.get_adjacent_traffic_lights(self.tls_id)

    def calculate_reward(self, state: Dict[str, Any], action: Optional[Tuple[int, str]], 
                         next_state: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Calculate the reward for taking a lane-level action.

        This default implementation provides a baseline reward function for
        lane-level control that can be overridden by subclasses.

        Args:
            state: Previous state dictionary with link_states and current_signal_state
            action: Action that was taken as (link_index, new_state)
            next_state: Resulting state dictionary

        Returns:
            reward: Calculated reward value
            components: Dictionary of reward components (for analysis)
        """
        if action is None:
            return 0.0, {}  # No action, no reward
            
        # Handle legacy state format for backward compatibility
        if not isinstance(state, dict) or not isinstance(next_state, dict):
            return self._legacy_calculate_reward(state, action, next_state)
            
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
        
    def _legacy_calculate_reward(self, state, action, next_state):
        """Legacy reward calculation for backward compatibility with phase-based control.

        Args:
            state: Previous state vector
            action: Action that was taken
            next_state: Resulting state vector

        Returns:
            reward: Calculated reward value
            components: Dictionary of reward components
        """
        # Get traffic light data
        tls_id = self.tls_id
        lanes = self.network.get_controlled_lanes(tls_id)

        # Extract state features
        waiting_time = next_state[0]
        vehicle_count = next_state[1]
        queue_length = next_state[2]
        throughput = next_state[3]

        # Get per-lane metrics
        lane_queues = [self.network.get_lane_queue(lane) for lane in lanes]
        lane_waiting = [self.network.get_lane_waiting_time(lane) for lane in lanes]

        # Default normalization constants
        MAX_WAITING_TIME = 500.0
        MAX_THROUGHPUT = 25.0
        MAX_QUEUE_LENGTH = 20.0
        MAX_VEHICLES = 40.0

        # Default component weights
        W_WAITING = 0.25
        W_THROUGHPUT = 0.45
        W_BALANCE = 0.15
        W_CHANGE = 0.05
        W_CONGESTION = 0.10

        # Normalize metrics
        import numpy as np
        norm_waiting = min(1.0, waiting_time / MAX_WAITING_TIME)
        norm_throughput = min(1.0, throughput / MAX_THROUGHPUT)
        norm_queue = min(1.0, sum(lane_queues) / MAX_QUEUE_LENGTH)
        norm_vehicles = min(1.0, vehicle_count / MAX_VEHICLES)

        # Calculate queue imbalance
        if sum(lane_queues) > 0:
            cv_queue = np.std(lane_queues) / (np.mean(lane_queues) + 1e-6)
            norm_imbalance = min(1.0, cv_queue)
        else:
            norm_imbalance = 0.0

        # Phase change penalty - agents should track their last action
        last_action = getattr(self, 'last_action', None)
        phase_change = 1.0 if action != last_action else 0.0
        self.last_action = action

        # Calculate components
        waiting_penalty = -norm_waiting * W_WAITING
        throughput_reward = norm_throughput * W_THROUGHPUT
        balance_penalty = -norm_imbalance * W_BALANCE
        change_penalty = -phase_change * W_CHANGE
        congestion_penalty = -norm_vehicles * W_CONGESTION

        # Combine components
        total_reward = waiting_penalty + throughput_reward + balance_penalty + change_penalty + congestion_penalty

        # Return reward and components for analysis
        components = {
            'waiting_penalty': waiting_penalty,
            'throughput_reward': throughput_reward,
            'balance_penalty': balance_penalty,
            'change_penalty': change_penalty,
            'congestion_penalty': congestion_penalty
        }

        return total_reward, components