from abc import ABC, abstractmethod
from typing import Dict, Any, ClassVar, List, Tuple, Optional
import os # <-- Add import

class Agent(ABC):
    """Abstract base class for all agents.
    # ... (rest of the docstring) ...
    """

    # Default configurations for agent subclasses
    DEFAULT_CONFIG: ClassVar[Dict[str, Any]] = {}

    def __init__(self, tls_id, network):
        """Initialize the agent.
        # ... (rest of the docstring) ...
        """
        self.tls_id = tls_id
        self.network = network
        self.last_action = None

    @classmethod
    def create(cls, tls_id, network, **kwargs):
        """Create an instance of the agent with proper configuration.
        # ... (rest of the docstring) ...
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
        # ... (rest of the docstring) ...
        """
        pass

    @abstractmethod
    def learn(self, state: Dict[str, Any], action: Optional[Tuple[int, str]],
              next_state: Dict[str, Any], done: bool) -> None:
        """Learn from experience.
        # ... (rest of the docstring) ...
        """
        pass

    # --- Add save_state and load_state ---
    @abstractmethod
    def save_state(self, directory_path: str):
        """
        Saves the agent's internal state (model weights, hyperparameters, etc.)
        to the specified directory. Implementations should ensure the directory
        exists.

        Args:
            directory_path: The path to the directory where state files should be saved.
        """
        # Base implementation can ensure the directory exists
        try:
            os.makedirs(directory_path, exist_ok=True)
        except OSError as e:
            # Log or handle the error appropriately if directory creation fails
            print(f"Error creating directory {directory_path}: {e}")
            # Depending on severity, you might want to raise the exception
            # raise
        pass # Concrete implementations will do the actual saving

    @abstractmethod
    def load_state(self, directory_path: str):
        """
        Loads the agent's internal state from the specified directory.
        Implementations should handle cases where files might be missing
        or corrupted.

        Args:
            directory_path: The path to the directory from which state files should be loaded.
        """
        pass
    # --- End of added methods ---

    def get_adjacent_traffic_lights(self, network) -> List[str]:
        """Determine adjacent traffic lights based on network topology.
        # ... (rest of the docstring) ...
        """
        return network.get_adjacent_traffic_lights(self.tls_id)

    def calculate_reward(self, state: Dict[str, Any], action: Optional[Tuple[int, str]],
                         next_state: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Calculate the reward for taking a lane-level action.
        # ... (rest of the implementation) ...
        """
        if action is None:
            return 0.0, {}  # No action, no reward

        # Handle legacy state format for backward compatibility
        if not isinstance(state, dict) or not isinstance(next_state, dict):
            return self._legacy_calculate_reward(state, action, next_state)

        # Ensure state dictionaries have the expected keys
        if 'link_states' not in next_state or 'link_states' not in state:
             # Handle cases where state might be incomplete, perhaps from early steps or errors
             # Return a default reward or log a warning
             # print(f"Warning: Incomplete state received in calculate_reward for {self.tls_id}")
             return 0.0, {}

        # Proceed only if action is valid tuple
        if not (isinstance(action, tuple) and len(action) == 2):
             # print(f"Warning: Invalid action format received in calculate_reward for {self.tls_id}: {action}")
             return 0.0, {}

        link_index, new_state = action

        # Get metrics for all links
        next_link_states = next_state.get('link_states', []) # Use .get for safety

        # Find the specific link that was changed
        target_link = None
        for link in next_link_states:
            if link.get('index') == link_index: # Use .get for safety
                target_link = link
                break

        if not target_link:
            # print(f"Warning: Target link {link_index} not found in next_state for {self.tls_id}")
            return 0.0, {}  # Link not found

        # Metrics for the targeted link (use .get with defaults)
        target_waiting_time = target_link.get('waiting_time', 0.0)
        target_queue_length = target_link.get('queue_length', 0)

        # Overall metrics across all links
        total_waiting_time = sum(link.get('waiting_time', 0.0) for link in next_link_states)
        max_queue_length = max((link.get('queue_length', 0) for link in next_link_states), default=0)
        # Use network reference if available and method exists
        total_throughput = 0
        if hasattr(self, 'network') and callable(getattr(self.network, 'get_departed_vehicles_count', None)):
             total_throughput = self.network.get_departed_vehicles_count()

        # Normalization constants
        MAX_WAITING_TIME = 500.0
        MAX_QUEUE_LENGTH = 20.0
        MAX_THROUGHPUT = 25.0

        # Normalize metrics safely
        norm_target_waiting = min(1.0, target_waiting_time / MAX_WAITING_TIME) if MAX_WAITING_TIME > 0 else 0.0
        num_links = len(next_link_states) if next_link_states else 1 # Avoid division by zero
        norm_total_waiting = min(1.0, total_waiting_time / (MAX_WAITING_TIME * num_links)) if MAX_WAITING_TIME > 0 else 0.0
        norm_max_queue = min(1.0, max_queue_length / MAX_QUEUE_LENGTH) if MAX_QUEUE_LENGTH > 0 else 0.0
        norm_throughput = min(1.0, total_throughput / MAX_THROUGHPUT) if MAX_THROUGHPUT > 0 else 0.0

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
        # ... (rest of the implementation) ...
        """
        # --- (Existing legacy code remains here) ---
        # Get traffic light data
        tls_id = self.tls_id
        lanes = self.network.get_controlled_lanes(tls_id)

        # Extract state features (ensure state is indexable)
        if not isinstance(state, (list, tuple)) or len(state) < 4 or \
           not isinstance(next_state, (list, tuple)) or len(next_state) < 4:
             print(f"Warning: Invalid legacy state format in _legacy_calculate_reward for {tls_id}")
             return 0.0, {}

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
        norm_waiting = min(1.0, waiting_time / MAX_WAITING_TIME) if MAX_WAITING_TIME > 0 else 0.0
        norm_throughput = min(1.0, throughput / MAX_THROUGHPUT) if MAX_THROUGHPUT > 0 else 0.0
        norm_queue = min(1.0, sum(lane_queues) / MAX_QUEUE_LENGTH) if MAX_QUEUE_LENGTH > 0 else 0.0
        norm_vehicles = min(1.0, vehicle_count / MAX_VEHICLES) if MAX_VEHICLES > 0 else 0.0

        # Calculate queue imbalance
        if sum(lane_queues) > 0:
            mean_queue = np.mean(lane_queues)
            if mean_queue > 1e-6: # Avoid division by zero
                 cv_queue = np.std(lane_queues) / mean_queue
                 norm_imbalance = min(1.0, cv_queue)
            else:
                 norm_imbalance = 0.0
        else:
            norm_imbalance = 0.0

        # Phase change penalty - agents should track their last action
        last_action = getattr(self, 'last_action', None)
        phase_change = 1.0 if action != last_action else 0.0
        # Update last action only if it's part of the logic, might belong in learn/choose_action
        # self.last_action = action # Reconsider if this is the right place

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