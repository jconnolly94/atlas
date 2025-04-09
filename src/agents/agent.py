# src/agents/agent.py

from abc import ABC, abstractmethod
from typing import Dict, Any, ClassVar, List, Tuple, Optional, Union
import os
import logging # Use logging instead of print for warnings/errors
import numpy as np # Needed for default reward calculation

# Configure a logger for this base class
agent_logger = logging.getLogger(__name__)
if not agent_logger.hasHandlers(): # Avoid adding handlers multiple times
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Agent(ABC):
    """
    Abstract base class for all traffic signal control agents.

    Defines the common interface for agents, including choosing actions,
    learning from experience, calculating rewards, and managing state.
    """

    # Default configurations for agent subclasses
    DEFAULT_CONFIG: ClassVar[Dict[str, Any]] = {}

    def __init__(self, tls_id: str, network: Any):
        """Initialize the agent.

        Args:
            tls_id: The identifier of the traffic light system this agent controls.
            network: An object providing access to the simulation network interface
                     (e.g., methods to get lane data, set phases).
        """
        self.tls_id = tls_id
        self.network = network
        # last_action stores the action taken in the *previous* step,
        # useful for calculating rewards or state features. Type depends on subclass.
        self.last_action: Optional[Union[int, Tuple[int, str]]] = None

    @classmethod
    def create(cls, tls_id: str, network: Any, **kwargs):
        """
        Factory class method to create an instance of the agent with merged configuration.

        Args:
            tls_id: The identifier of the traffic light system.
            network: The network interface object.
            **kwargs: Specific configuration parameters to override defaults.

        Returns:
            An instance of the concrete Agent subclass.
        """
        # Start with default configuration defined in the subclass
        config = cls.DEFAULT_CONFIG.copy()
        # Override defaults with any provided keyword arguments
        config.update(kwargs)
        # Create and return instance, passing the merged config
        return cls(tls_id, network, **config)

    @abstractmethod
    def choose_action(self, state: Dict[str, Any]) -> Optional[Union[int, Tuple[int, str]]]:
        """
        Choose an action based on the current state observation.

        The action format depends on the agent implementation:
        - Phase-based agents typically return an integer phase index: `Optional[int]`
        - Lane-level agents typically return a tuple: `Optional[Tuple[int, str]]`

        Args:
            state: A dictionary representing the current state of the environment
                   relevant to this agent (e.g., containing 'link_states',
                   'current_signal_state').

        Returns:
            The chosen action (format depends on agent type) or None if no action
            should be taken.
        """
        pass

    @abstractmethod
    def learn(self, state: Dict[str, Any], action: Optional[Union[int, Tuple[int, str]]],
              next_state: Dict[str, Any], done: bool) -> None:
        """
        Update the agent's knowledge or policy based on a transition.

        Args:
            state: The state before the action was taken.
            action: The action that was taken (format depends on agent type).
            next_state: The state observed after the action was taken.
            done: A boolean indicating whether the episode terminated after this transition.
        """
        pass

    @abstractmethod
    def save_state(self, directory_path: str):
        """
        Saves the agent's internal state (model weights, hyperparameters, etc.)
        to the specified directory. Implementations should ensure the directory
        exists.

        Args:
            directory_path: The path to the directory where state files should be saved.
        """
        # Base implementation ensures the directory exists
        try:
            os.makedirs(directory_path, exist_ok=True)
            # agent_logger.debug(f"Ensured directory exists: {directory_path}") # Optional debug log
        except OSError as e:
            agent_logger.error(f"Error creating directory {directory_path}: {e}")
            # Depending on severity, you might want to raise the exception
            raise
        # Concrete implementations will perform the actual saving logic after this.

    @abstractmethod
    def load_state(self, directory_path: str):
        """
        Loads the agent's internal state from the specified directory.
        Implementations should handle cases where files might be missing
        or corrupted gracefully.

        Args:
            directory_path: The path to the directory from which state files should be loaded.
        """
        pass

    def get_adjacent_traffic_lights(self, network: Any) -> List[str]:
        """
        Determine adjacent traffic lights based on network topology using the network object.

        Args:
            network: The network interface object.

        Returns:
            A list of IDs for adjacent traffic light systems.
        """
        # Check if the network object has the required method
        if hasattr(network, 'get_adjacent_traffic_lights') and callable(network.get_adjacent_traffic_lights):
            try:
                return network.get_adjacent_traffic_lights(self.tls_id)
            except Exception as e:
                agent_logger.error(f"Error calling network.get_adjacent_traffic_lights for {self.tls_id}: {e}")
                return [] # Return empty list on error
        else:
            agent_logger.warning("Network object does not have 'get_adjacent_traffic_lights' method.")
            return []

    def calculate_reward(self, state: Dict[str, Any], action: Optional[Union[int, Tuple[int, str]]],
                         next_state: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate a basic reward based on the change in total waiting time.
        Subclasses should override this method to implement more specific
        or sophisticated reward functions tailored to their control strategy
        (e.g., phase-based vs. lane-level).

        Args:
            state: The state before the action was taken.
            action: The action taken (phase index or lane-level tuple).
            next_state: The state observed after the action.

        Returns:
            A tuple containing:
                - reward (float): The calculated reward value.
                - components (Dict[str, float]): A dictionary detailing components
                  of the reward for analysis/logging (optional).
        """
        if action is None:
            return 0.0, {'reason': 'no_action'}

        # Ensure state dictionaries are valid and contain link_states
        if not isinstance(state, dict) or 'link_states' not in state or \
           not isinstance(next_state, dict) or 'link_states' not in next_state:
             agent_logger.warning(f"Incomplete state received in base calculate_reward for {self.tls_id}")
             return 0.0, {'reason': 'invalid_state'}

        prev_link_states = state.get('link_states', [])
        next_link_states = next_state.get('link_states', [])

        if not prev_link_states or not next_link_states:
            return 0.0, {'reason': 'missing_link_states'}

        # Calculate total waiting time change (reduction is positive reward)
        prev_total_wait = sum(link.get('waiting_time', 0.0) for link in prev_link_states)
        next_total_wait = sum(link.get('waiting_time', 0.0) for link in next_link_states)
        wait_time_reduction = prev_total_wait - next_total_wait

        # Basic reward scaling (adjust scale factor as needed)
        REWARD_SCALE = 100.0
        reward = wait_time_reduction / REWARD_SCALE

        # Clip reward to a reasonable range
        reward = max(-1.0, min(reward, 1.0))

        components = {
            'wait_time_reduction': wait_time_reduction,
            'scaled_reward': reward
        }

        return reward, components

    # --- Optional: Keep legacy reward calculation for specific debugging or comparison ---
    def _legacy_calculate_reward(self, state, action, next_state):
        """Legacy reward calculation for backward compatibility if needed."""
        # (Keep the original implementation from your file if required for specific agents)
        # For now, just return a default value.
        agent_logger.warning(f"Using fallback _legacy_calculate_reward for {self.tls_id}")
        return 0.0, {}