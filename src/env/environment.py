from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from src.utils.observer import Observable
from src.utils.data_collector import MetricsCalculator


class Environment(Observable):
    """Environment that manages the traffic simulation."""

    def __init__(self, network):
        """Initialize the environment.

        Args:
            network: Network object providing access to the traffic simulation
        """
        super().__init__()  # Initialize Observable base class
        self.network = network
        self.metrics_calculator = MetricsCalculator(network)
        self.current_states = {}
        self.episode_step = 0
        self.episode_number = 0
        self.episode_metrics = {
            'waiting_times': [],
            'rewards': [],
            'throughput': []
        }
        # Keep track of last actions for all traffic lights
        self.last_actions = {}

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset the environment for a new episode.

        Returns:
            Dictionary mapping traffic light IDs to initial states
        """
        # Reset the simulation
        self.network.reset_simulation()

        # Reset environment tracking variables
        self.current_states = {}
        self.last_actions = {}  # Reset last actions
        self.episode_step = 0
        self.episode_number += 1
        self.episode_metrics = {
            'waiting_times': [],
            'rewards': [],
            'throughput': []
        }

        # Get initial state
        self.current_states = self.network.get_state()

        # Initialize last actions for all traffic lights with safe default
        for tls_id in self.network.tls_ids:
            self.last_actions[tls_id] = 0  # Default to phase 0

        # Ensure network is properly updated after reset
        self.network.update_edge_data()
        self.network.update_arrivals()

        return self.current_states

    def get_state(self) -> Dict[str, np.ndarray]:
        """Get the current state of the environment.

        Returns:
            Dictionary mapping traffic light IDs to state vectors
        """
        self.current_states = self.network.get_state()
        return self.current_states

    def step(self, agents: Dict[str, 'Agent']) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool]:
        """Perform one step in the environment with the given agents.

        Args:
            agents: Dictionary mapping traffic light IDs to agent objects

        Returns:
            next_states: Dictionary mapping traffic light IDs to next states
            rewards: Dictionary mapping traffic light IDs to rewards
            done: Whether the episode is complete
        """
        # Record old states for reward calculation
        old_states = self.current_states.copy()

        # Choose actions for each agent
        actions = {}
        for tls_id, agent in agents.items():
            state = old_states.get(tls_id)
            if state is not None:
                action = agent.choose_action(state)
                actions[tls_id] = action
                # Update last_actions even if None
                if action is not None:
                    self.last_actions[tls_id] = action

        # Apply actions
        self.apply_actions(actions)

        # Get new states
        next_states = self.get_state()

        # Calculate rewards and let agents learn
        rewards = {}
        total_waiting_time = 0

        for tls_id, agent in agents.items():
            # Skip if we don't have both states
            if tls_id not in old_states or tls_id not in next_states:
                continue

            old_state = old_states[tls_id]
            next_state = next_states[tls_id]
            action = actions.get(tls_id)

            # If action is None, use last known action for this TLS
            effective_action = action if action is not None else self.last_actions.get(tls_id, 0)

            # Calculate reward using the agent's own reward function
            reward, components = agent.calculate_reward(old_state, effective_action, next_state)
            rewards[tls_id] = reward

            # Track metrics for this step
            done = self.is_terminal_state()
            agent.learn(old_state, effective_action, next_state, done)

            # Update metrics - use effective_action for data collection
            metrics = self.metrics_calculator.calculate_step_metrics(tls_id, effective_action, reward)
            total_waiting_time += metrics['waiting_time']

            # Notify observers about step completion
            self.notify_step_complete({
                'episode': self.episode_number,
                'step': self.episode_step,
                **metrics
            })

        # Track episode metrics
        self.episode_metrics['waiting_times'].append(total_waiting_time)
        self.episode_metrics['rewards'].append(sum(rewards.values()))
        self.episode_metrics['throughput'].append(self.network.get_departed_vehicles_count())

        # Update step counter
        self.episode_step += 1

        # Check if episode is complete
        done = self.is_terminal_state()

        # If episode is complete, notify observers
        if done:
            self.notify_episode_complete({
                'episode': self.episode_number,
                'avg_waiting': np.mean(self.episode_metrics['waiting_times']) if self.episode_metrics[
                    'waiting_times'] else 0,
                'total_reward': sum(self.episode_metrics['rewards']),
                'arrived_vehicles': self.network.get_arrived_vehicles_count(),
                'total_steps': self.episode_step
            })

        return next_states, rewards, done

    def apply_actions(self, actions: Dict[str, Any]) -> None:
        """Apply actions to the traffic network.

        Args:
            actions: Dictionary mapping traffic light IDs to actions
        """
        for tls_id, action in actions.items():
            # Skip if action is None (no-op)
            if action is None:
                continue

            # Handle both simple actions and tuple actions (phase, duration)
            if isinstance(action, tuple):
                phase, duration = action
                self.network.set_traffic_light_phase(tls_id, phase)
                self.network.set_phase_duration(tls_id, duration)
            else:
                # For simple phase actions
                self.network.set_traffic_light_phase(tls_id, action)

        # Run simulation for specified number of steps
        for _ in range(self.network.simulation_steps_per_action):
            self.network.simulation_step()

        # Update network state
        self.network.update_edge_data()
        self.network.update_arrivals()

    def is_terminal_state(self) -> bool:
        """Check if the current state is terminal.

        Returns:
            True if the episode is complete, False otherwise
        """
        return self.network.is_simulation_complete()