from abc import ABC, abstractmethod
from typing import Dict, Any, ClassVar, List


class Agent(ABC):
    """Abstract base class for all agents.

    Each agent implementation should handle its own:
    1. State processing
    2. Action selection
    3. Learning mechanism
    4. Reward calculation
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
    def choose_action(self, state):
        """Choose an action based on the current state.

        Args:
            state: Current state observation

        Returns:
            The selected action
        """
        pass

    @abstractmethod
    def learn(self, state, action, next_state, done):
        """Learn from experience.

        Args:
            state: Previous state
            action: Action that was taken
            next_state: Resulting state
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

    def calculate_reward(self, state, action, next_state):
        """Calculate the reward for taking an action.

        This default implementation can be overridden by subclasses
        to provide agent-specific reward functions.

        Args:
            state: Previous state
            action: Action that was taken
            next_state: Resulting state

        Returns:
            reward: Calculated reward value
            components: Dictionary of reward components (for analysis)
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

        # Default normalization constants - increased for more headroom
        MAX_WAITING_TIME = 500.0  # Increased to be less punishing for high waiting times
        MAX_THROUGHPUT = 25.0     # Increased to reward throughput more
        MAX_QUEUE_LENGTH = 20.0   # Increased to be less punishing for queues
        MAX_VEHICLES = 40.0       # Increased to be less punishing for congestion

        # Default component weights - rebalanced to emphasize throughput more than waiting time
        W_WAITING = 0.25          # Reduced weight for waiting time penalty (was 0.40)
        W_THROUGHPUT = 0.45       # Increased weight for throughput reward (was 0.30)
        W_BALANCE = 0.15          # Lane balance (unchanged)
        W_CHANGE = 0.05           # Reduced phase change penalty (was 0.10)
        W_CONGESTION = 0.10       # Increased congestion weight to better handle traffic

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