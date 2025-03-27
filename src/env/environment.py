from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from src.utils.observer import Observable
from src.utils.data_collector import MetricsCalculator


class Environment(Observable):
    """Environment that manages the traffic simulation with lane-level traffic control."""

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
        # Track time since last change for each link
        self.last_change_time = {}
        # Track current simulation time
        self.current_time = 0.0

    def reset(self) -> Dict[str, Any]:
        """Reset the environment for a new episode.

        Returns:
            Dictionary mapping traffic light IDs to initial states
        """
        # Reset the simulation
        self.network.reset_simulation()

        # Reset environment tracking variables
        self.current_states = {}
        self.last_actions = {}  # Reset last actions
        self.last_change_time = {}  # Reset last change time
        self.episode_step = 0
        self.episode_number += 1
        self.episode_metrics = {
            'waiting_times': [],
            'rewards': [],
            'throughput': []
        }
        self.current_time = self.network.get_current_time()

        # Get initial state
        self.current_states = self.get_state()

        # Initialize last change time for each traffic light and its links
        for tls_id in self.network.tls_ids:
            # Initialize with empty dict - no action history yet
            self.last_actions[tls_id] = None
            
            # Initialize last change time for each traffic light and its links
            signal_state = self.network.get_red_yellow_green_state(tls_id)
            self.last_change_time[tls_id] = {}
            for i in range(len(signal_state)):
                self.last_change_time[tls_id][i] = self.current_time

        # Ensure network is properly updated after reset
        self.network.update_edge_data()
        self.network.update_arrivals()

        return self.current_states

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the environment with detailed link-level information.

        Returns:
            Dictionary mapping traffic light IDs to state information
        """
        states = {}
        self.current_time = self.network.get_current_time()

        for tls_id in self.network.tls_ids:
            # Get link-specific metrics
            link_metrics = self.network.get_link_metrics(tls_id)
            
            # Get current signal state
            current_signal_state = self.network.get_red_yellow_green_state(tls_id)
            
            # Initialize last_change_time dictionary for this TLS if it doesn't exist yet
            if tls_id not in self.last_change_time:
                self.last_change_time[tls_id] = {}
                for i in range(len(current_signal_state)):
                    self.last_change_time[tls_id][i] = self.current_time
            
            # Create an observation vector for each link
            link_states = []
            for link in link_metrics:
                link_index = link['index']
                
                # Ensure the link index exists in last_change_time dict
                if link_index not in self.last_change_time[tls_id]:
                    self.last_change_time[tls_id][link_index] = self.current_time
                    
                # Calculate time since last change
                time_since_change = self.current_time - self.last_change_time[tls_id][link_index]
                
                link_states.append({
                    'index': link_index,
                    'waiting_time': link['waiting_time'],
                    'queue_length': link['queue_length'],
                    'vehicle_count': link['vehicle_count'],
                    'current_state': current_signal_state[link_index],
                    # Time data
                    'time_since_last_change': time_since_change,
                    'current_simulation_time': self.current_time
                })
            
            # Store the complete lane-level state
            states[tls_id] = {
                'link_states': link_states,
                'current_signal_state': current_signal_state,
            }

        self.current_states = states
        return states

    def step(self, agents: Dict[str, 'Agent']) -> Tuple[Dict[str, Any], Dict[str, float], bool]:
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
            effective_action = action if action is not None else self.last_actions.get(tls_id)

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
        """Apply lane-level actions to the traffic network.
        
        Args:
            actions: Dictionary mapping traffic light IDs to lane-level actions
        """
        # Track delayed actions (for yellow → red transitions)
        delayed_actions = {}
        
        for tls_id, action in actions.items():
            # Skip if action is None (no-op)
            if action is None:
                continue

            # Only accept lane-level control actions
            if isinstance(action, tuple) and len(action) == 2 and isinstance(action[0], int) and isinstance(action[1], str):
                # Lane-level action format (link_index, new_state)
                link_index, new_state = action
                result = self.network.change_specific_link(tls_id, link_index, new_state)
                
                # Handle yellow transition scheduling if needed
                if isinstance(result, dict) and result.get('needs_followup'):
                    delayed_actions[tls_id] = {
                        'link_index': result['link_index'],
                        'next_state': result['next_state'],
                        'steps_remaining': 3  # Default yellow time in steps
                    }
                elif result:
                    # Track applied actions and update last change time
                    self.last_actions[tls_id] = action
                    self.last_change_time[tls_id][link_index] = self.network.get_current_time()
            else:
                # Invalid action format - only lane-level control actions are supported
                print(f"Warning: Invalid action format for {tls_id}: {action}")
                print(f"Only lane-level control actions (link_index, state_string) are supported.")

        # Run simulation for specified number of steps
        for _ in range(self.network.simulation_steps_per_action):
            # Process delayed actions
            for tls_id, delayed in list(delayed_actions.items()):
                delayed['steps_remaining'] -= 1
                
                if delayed['steps_remaining'] <= 0:
                    # Complete the delayed transition
                    link_index = delayed['link_index']
                    next_state = delayed['next_state']
                    
                    # Apply the change directly without further checks
                    current_state = self.network.get_red_yellow_green_state(tls_id)
                    state_list = list(current_state)
                    state_list[link_index] = next_state
                    self.network.set_red_yellow_green_state(tls_id, ''.join(state_list))
                    
                    # Update last action and last change time, then remove from delayed
                    self.last_actions[tls_id] = (link_index, next_state)
                    self.last_change_time[tls_id][link_index] = self.network.get_current_time()
                    del delayed_actions[tls_id]
            
            # Step the simulation
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
    
    def get_time_since_last_change(self, tls_id: str, link_index: int) -> float:
        """Get the time since the last state change for a specific link.
        
        Args:
            tls_id: ID of the traffic light
            link_index: Index of the link
            
        Returns:
            Time in seconds since the last state change
        """
        if tls_id not in self.last_change_time or link_index not in self.last_change_time[tls_id]:
            return 0.0
            
        return self.current_time - self.last_change_time[tls_id][link_index]