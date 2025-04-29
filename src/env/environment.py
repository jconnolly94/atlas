# src/env/environment.py

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import logging

from src.utils.observer import Observable
from src.utils.data_collector import MetricsCalculator

class Environment(Observable):
    """Environment that manages the traffic simulation, supporting phase-based actions."""

    def __init__(self, network, data_queue=None):
        """Initialize the environment.

        Args:
            network: Network object providing access to the traffic simulation
            data_queue: Optional multiprocessing queue for asynchronous data writing.
        """
        super().__init__()  # Initialize Observable base class
        self.network = network
        self.metrics_calculator = MetricsCalculator(network)
        self.data_queue = data_queue  # Store the queue
        self.current_states = {}
        self.episode_step = 0
        self.episode_number = 0
        self.last_termination_reason = None  # Add this to store reason
        self.last_episode_step_count = 0  # Add this
        self.episode_metrics = {
            'waiting_times': [],
            'rewards': [],
            'throughput': []
        }
        # Keep track of last actions (now phase indices) for all traffic lights
        self.last_actions: Dict[str, Optional[int]] = {}
        # Track time since last change for each link (still useful for state/reward)
        self.last_change_time: Dict[str, Dict[int, float]] = {}
        # Track current simulation time
        self.current_time = 0.0

        # Configure logging for this module
        self.logger = logging.getLogger(f"Environment_{id(self)}") # Unique logger instance
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            # Add handler if none exist (e.g., running standalone)
             handler = logging.StreamHandler()
             formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
             handler.setFormatter(formatter)
             self.logger.addHandler(handler)


    def reset(self) -> Dict[str, Any]:
        """Reset the environment for a new episode.

        Returns:
            Dictionary mapping traffic light IDs to initial states
        """
        self.logger.info(f"Resetting environment for episode {self.episode_number + 1}")
        # Reset the simulation
        self.network.reset_simulation()

        # Reset environment tracking variables
        self.current_states = {}
        self.last_actions = {}  # Reset last actions
        self.last_change_time = {}  # Reset last change time
        self.episode_step = 0
        self.episode_number += 1
        self.last_termination_reason = None  # Clear reason on reset
        self.last_episode_step_count = 0
        self.episode_metrics = {
            'waiting_times': [],
            'rewards': [],
            'throughput': []
        }
        self.current_time = self.network.get_current_time()

        # Get initial state
        self.current_states = self.get_state() # get_state needs to handle initialization correctly

        # Initialize last change time for each traffic light and its links
        for tls_id in self.network.tls_ids:
            # Initialize last action for this TLS
            self.last_actions[tls_id] = None

            # Get initial signal state (may be empty string immediately after reset)
            signal_state = self.network.get_red_yellow_green_state(tls_id)
            self.last_change_time[tls_id] = {}
            if signal_state: # Only initialize if signal state is available
                for i in range(len(signal_state)):
                    self.last_change_time[tls_id][i] = self.current_time
            else:
                self.logger.warning(f"Could not get initial signal state for {tls_id} during reset.")


        # Ensure network is properly updated after reset
        # self.network.update_edge_data() # May not be needed if get_state reads fresh
        self.network.update_arrivals()
        return self.current_states

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the environment with detailed link-level information.
           (Still provides link-level info which can be used by agents for state processing).

        Returns:
            Dictionary mapping traffic light IDs to state information
        """
        states = {}
        try:
            self.current_time = self.network.get_current_time()
        except Exception as e:
             self.logger.error(f"Error getting current time: {e}. Returning empty state.")
             return {} # Cannot proceed if simulation connection is lost

        for tls_id in self.network.tls_ids:
            try:
                # Get link-specific metrics
                link_metrics = self.network.get_link_metrics(tls_id)

                # Get current signal state
                current_signal_state = self.network.get_red_yellow_green_state(tls_id)
                if not current_signal_state:
                    self.logger.warning(f"Could not get signal state for {tls_id}. Skipping state update.")
                    continue # Skip this TLS if state unavailable

                # Initialize last_change_time dictionary for this TLS if it doesn't exist yet
                # This might happen if a new TLS appears or after reset issues
                if tls_id not in self.last_change_time:
                    self.last_change_time[tls_id] = {}
                    for i in range(len(current_signal_state)):
                        self.last_change_time[tls_id][i] = self.current_time

                # Create an observation vector for each link
                link_states_list = []
                for link in link_metrics:
                    link_index = link['index']

                    # Ensure the link index exists in last_change_time dict
                    # This handles cases where link structure might change dynamically (unlikely)
                    # or initialization gaps.
                    if link_index not in self.last_change_time[tls_id]:
                        self.last_change_time[tls_id][link_index] = self.current_time

                    # Calculate time since last change for this specific link
                    time_since_change = self.current_time - self.last_change_time[tls_id][link_index]

                    link_states_list.append({
                        'index': link_index,
                        'waiting_time': link.get('waiting_time', 0.0), # Use .get for safety
                        'queue_length': link.get('queue_length', 0),
                        'vehicle_count': link.get('vehicle_count', 0),
                        'current_state': current_signal_state[link_index] if link_index < len(current_signal_state) else '?', # Handle index out of bounds
                        'time_since_last_change': time_since_change,
                        'current_simulation_time': self.current_time
                    })

                # Store the complete lane-level state
                states[tls_id] = {
                    'link_states': link_states_list,
                    'current_signal_state': current_signal_state,
                    # Agents might also need the current phase index
                    'current_phase_index': self.network.get_current_phase_index(tls_id) # Use network interface
                }
            except Exception as e:
                 self.logger.error(f"Error getting state for TLS {tls_id}: {e}. Skipping this TLS.", exc_info=True)
                 continue # Skip this TLS if error occurs

        self.current_states = states
        return states

    def step(self, agents: Dict[str, 'Agent'], agent_type: str = None, network_name: str = None, early_term_config: Dict = None) -> Tuple[Dict[str, Any], Dict[str, float], bool]:
        """Perform one step in the environment using actions from agents.

        Args:
            agents: Dictionary mapping traffic light IDs to agent objects.
            agent_type: Type of the agent (for data logging).
            network_name: Name of the network (for data logging).
            early_term_config: Configuration for early episode termination.

        Returns:
            Tuple containing:
                - next_states: Dictionary mapping traffic light IDs to next states.
                - rewards: Dictionary mapping traffic light IDs to rewards.
                - done: Boolean indicating if the episode is complete.
        """
        if not self.current_states:
             self.logger.warning("Step called with no current state. Attempting to get state.")
             self.current_states = self.get_state()
             if not self.current_states:
                 self.logger.error("Failed to get state in step(). Returning empty results.")
                 return {}, {}, True # End episode if state cannot be retrieved

        old_states = self.current_states.copy() # Use the most recent valid state

        # 1. Choose actions for each agent
        actions = {}
        for tls_id, agent in agents.items():
            state = old_states.get(tls_id)
            if state is not None:
                try:
                    # Agent chooses an action (now expecting phase index)
                    action = agent.choose_action(state)
                    actions[tls_id] = action
                    # Update last_actions (stores chosen phase index)
                    if action is not None:
                        self.last_actions[tls_id] = action
                except Exception as e:
                    self.logger.error(f"Error getting action from agent {tls_id} ({type(agent).__name__}): {e}", exc_info=True)
                    actions[tls_id] = None # Agent failed, take no action
            else:
                self.logger.warning(f"No state found for agent {tls_id} in step().")
                actions[tls_id] = None

        # 2. Apply actions to the simulation
        self.apply_actions(actions) # This now handles phase indices

        # 3. Get new states after simulation steps
        next_states = self.get_state()

        # Check for simulation connection issues after stepping
        if not next_states and self.network.tls_ids: # If state is empty but we expect TLS
             self.logger.error("Failed to get next state after applying actions. Simulation likely disconnected. Ending episode.")
             return {}, {}, True # End episode


        # 4. Check for simulation completion (default SUMO condition)
        done = self.is_terminal_state()
        early_termination_reason = None


        # 5. Check for early termination conditions (based on metrics)
        if not done and early_term_config and early_term_config.get('enabled', False) and \
           self.episode_step >= early_term_config.get('min_steps_before_check', 100): # Ensure min_steps is non-negative

            max_wait = 0.0
            max_queue = 0
            # Use next_states for checking termination conditions
            for tls_id, state_data in next_states.items():
                 # Check if state_data is valid and has 'link_states'
                 if isinstance(state_data, dict) and 'link_states' in state_data:
                     for link_state in state_data['link_states']:
                         if isinstance(link_state, dict): # Check link_state structure
                             max_wait = max(max_wait, link_state.get('waiting_time', 0.0))
                             max_queue = max(max_queue, link_state.get('queue_length', 0))
                         else:
                              self.logger.warning(f"Invalid link_state format in early termination check: {link_state}")
                 else:
                     self.logger.warning(f"Invalid state_data format for TLS {tls_id} in early termination check: {state_data}")


            wait_threshold = early_term_config.get('max_step_wait_time', 300.0)
            queue_threshold = early_term_config.get('max_step_queue_length', 40)

            if max_wait > wait_threshold:
                # Create detailed message for logging
                detailed_reason = f"Max wait time exceeded ({max_wait:.1f}s > {wait_threshold:.1f}s)"
                # Use standardized category string for data storage
                early_termination_reason = "max_wait_time_exceeded"
                # Log the detailed message
                self.logger.warning(f"Early termination at episode {self.episode_number}, step {self.episode_step}: {detailed_reason}")
            elif max_queue > queue_threshold:
                # Create detailed message for logging
                detailed_reason = f"Max queue length exceeded ({max_queue} > {queue_threshold})"
                # Use standardized category string for data storage
                early_termination_reason = "max_queue_length_exceeded"
                # Log the detailed message
                self.logger.warning(f"Early termination at episode {self.episode_number}, step {self.episode_step}: {detailed_reason}")

            if early_termination_reason:
                done = True
                self.last_termination_reason = early_termination_reason


        # 6. Calculate rewards
        final_rewards = {}
        total_step_reward = 0.0 # Aggregate reward for logging/analysis if needed

        for tls_id, agent in agents.items():
            old_state = old_states.get(tls_id)
            next_state = next_states.get(tls_id) # Use the potentially empty next_state
            action = actions.get(tls_id) # The chosen action (phase index or None)

            # Use last_action if current action is None (agent might not have acted)
            effective_action = action if action is not None else self.last_actions.get(tls_id)

            # Calculate reward only if both states are valid
            if old_state is not None and next_state is not None:
                try:
                    # Agent calculates reward based on state transition
                    # Pass the phase index as the action
                    reward, components = agent.calculate_reward(old_state, effective_action, next_state)

                    # Apply termination penalty if applicable
                    if early_termination_reason:
                        termination_penalty = early_term_config.get('termination_penalty', -100.0)
                        final_rewards[tls_id] = termination_penalty
                    else:
                        final_rewards[tls_id] = reward

                    total_step_reward += final_rewards[tls_id]

                except Exception as e:
                    self.logger.error(f"Error calculating reward for agent {tls_id}: {e}", exc_info=True)
                    final_rewards[tls_id] = 0.0 # Default reward on error
            else:
                 # Handle cases where state might be missing (e.g., simulation issues)
                 final_rewards[tls_id] = 0.0
                 if old_state is None: self.logger.warning(f"Missing old_state for {tls_id} in reward calculation.")
                 if next_state is None: self.logger.warning(f"Missing next_state for {tls_id} in reward calculation.")


        # 7. Let agents learn
        for tls_id, agent in agents.items():
            old_state = old_states.get(tls_id)
            next_state = next_states.get(tls_id)
            action = actions.get(tls_id) # Use the action chosen for this step
            reward = final_rewards.get(tls_id, 0.0) # Get the final reward

            # Agent learns only if states are valid
            if old_state is not None and next_state is not None:
                try:
                    # Pass the chosen phase index (or None) to learn
                    agent.learn(old_state, action, next_state, done)
                except Exception as e:
                    self.logger.error(f"Error during learning for agent {tls_id}: {e}", exc_info=True)

        # 8. Collect and Log Step Metrics (using Data Queue)
        step_metrics_collected = False
        if self.data_queue:
             for tls_id, reward in final_rewards.items():
                 effective_action = actions.get(tls_id) # Get the action chosen for this step
                 next_state = next_states.get(tls_id) # Get the resulting state

                 # Calculate metrics only if next_state is valid
                 if next_state is not None and isinstance(next_state, dict) and 'link_states' in next_state:
                     link_states = next_state['link_states']
                     waiting_time = sum(link.get('waiting_time', 0.0) for link in link_states)
                     vehicle_count = sum(link.get('vehicle_count', 0) for link in link_states)
                     queue_length = sum(link.get('queue_length', 0) for link in link_states)

                     step_data_dict = {
                         'type': 'step', # Crucial for data_writer_process
                         'agent_type': agent_type or 'Unknown',
                         'network': network_name or 'Unknown',
                         'episode': self.episode_number,
                         'step': self.episode_step,
                         'tls_id': tls_id,
                         'action': str(effective_action), # Store action (phase index or None) as string
                         'reward': float(reward),
                         'waiting_time': float(waiting_time),
                         'vehicle_count': int(vehicle_count),
                         'queue_length': int(queue_length)
                     }
                     try:
                         self.data_queue.put(step_data_dict)
                         step_metrics_collected = True
                     except Exception as q_err:
                          self.logger.error(f"Failed to put step data on queue: {q_err}")
                 else:
                      self.logger.warning(f"Skipping step metrics for {tls_id} due to invalid next_state.")

        # 9. Track Episode Metrics (using calculated step rewards)
        self.episode_metrics['waiting_times'].append(sum(link.get('waiting_time', 0.0)
                                                          for tls_id, state_data in next_states.items()
                                                          if isinstance(state_data, dict) and 'link_states' in state_data
                                                          for link in state_data['link_states']))
        self.episode_metrics['rewards'].append(sum(final_rewards.values()))
        try:
            self.episode_metrics['throughput'].append(self.network.get_departed_vehicles_count())
        except Exception as e:
            self.logger.warning(f"Error getting departed vehicles count: {e}")
            self.episode_metrics['throughput'].append(0) # Append default on error


        # 10. Update step counter
        self.episode_step += 1
        self.last_episode_step_count = self.episode_step

        # 11. Handle Episode Completion
        if done:
            # Determine final reason for termination
            if self.last_termination_reason:
                final_reason = self.last_termination_reason
            elif self.episode_step >= 1000: # Assuming max_steps = 1000 (should be configurable)
                 final_reason = 'max_steps'
            else:
                 final_reason = 'natural' # Simulation ended normally

            self.logger.info(f"Episode {self.episode_number} finished at step {self.last_episode_step_count}. Reason: {final_reason}")

            # Calculate final episode metrics
            avg_waiting = np.mean(self.episode_metrics['waiting_times']) if self.episode_metrics['waiting_times'] else 0
            total_reward = sum(self.episode_metrics['rewards']) if self.episode_metrics['rewards'] else 0
            try:
                 # Get final throughput (arrived vehicles)
                 final_throughput = self.network.get_arrived_vehicles_count()
            except Exception as e:
                 self.logger.error(f"Error getting arrived vehicles count at episode end: {e}")
                 final_throughput = -1 # Indicate error

            # Prepare episode data for logging/queue
            episode_summary_data = {
                'type': 'episode', # For data_writer_process
                'agent_type': agent_type or 'Unknown',
                'network': network_name or 'Unknown',
                'episode': self.episode_number,
                'avg_waiting': float(avg_waiting),
                'total_reward': float(total_reward),
                'total_steps': self.last_episode_step_count,
                'final_throughput': final_throughput,
                'termination_reason': final_reason
            }

            # Log episode summary
            self.logger.info(f"Episode {self.episode_number} Summary: AvgWait={avg_waiting:.2f}, "
                             f"TotalReward={total_reward:.2f}, Steps={self.last_episode_step_count}, "
                             f"Throughput={final_throughput}, Reason='{final_reason}'")

            # Put episode data onto the queue
            if self.data_queue:
                try:
                    self.data_queue.put(episode_summary_data)
                except Exception as q_err:
                    self.logger.error(f"Failed to put episode data on queue: {q_err}")

            # Notify legacy observers (if any)
            # Ensure data passed matches observer expectations if still used
            observer_data = {
                 'episode': self.episode_number,
                 'avg_waiting': avg_waiting,
                 'total_reward': total_reward,
                 'arrived_vehicles': final_throughput, # Legacy name
                 'total_steps': self.last_episode_step_count,
                 'termination_reason': final_reason
             }
            self.notify_episode_complete(observer_data)

        # Return the results of the step
        return next_states, final_rewards, done


    # --- apply_actions Modified for Phase Control ---
    def apply_actions(self, actions: Dict[str, Optional[int]]) -> None:
        """Apply phase-based actions to the traffic network.

        Args:
            actions: Dictionary mapping traffic light IDs to phase indices (int) or None.
        """
        applied_actions_count = 0
        for tls_id, phase_index in actions.items():
            # Skip if action is None (agent decided not to act or action was invalid)
            if phase_index is None:
                continue

            # Check if action is a valid phase index (integer)
            if isinstance(phase_index, int):
                try:
                    # Set the phase using network interface
                    # This should handle transitions including yellow lights
                    # based on the phase definitions in the .net.xml file.
                    self.network.set_traffic_light_phase(tls_id, phase_index)
                    applied_actions_count += 1
                    # self.logger.debug(f"Applied phase {phase_index} to TLS {tls_id}")

                    # Update last_action for the agent's internal tracking (if needed elsewhere)
                    # self.last_actions[tls_id] = phase_index # Already updated in step()

                    # --- Update last_change_time logic for phase changes ---
                    # When a phase changes, potentially multiple link states change.
                    # We might need to query the new RYG state and update all link change times.
                    # For simplicity now, we might skip detailed link-level time tracking
                    # if the state representation mainly relies on phase index and aggregate metrics.
                    # If `time_since_last_change` per link is crucial for the agent's state,
                    # we need to add logic here to update self.last_change_time accordingly.
                    # Example (Needs refinement):
                    # new_ryg_state = self.network.get_red_yellow_green_state(tls_id)
                    # if tls_id in self.last_change_time:
                    #     for i in range(len(new_ryg_state)):
                    #         # Simplistic: update time for all links on phase change
                    #         self.last_change_time[tls_id][i] = self.network.get_current_time()

                except KeyError:
                     self.logger.error(f"Environment: Invalid TLS ID '{tls_id}' provided in actions.")
                except Exception as e: # Catch other potential errors
                    self.logger.warning(f"Environment: Could not set phase {phase_index} for TLS {tls_id}: {e}")
            else:
                # Handle unexpected action format
                 self.logger.warning(f"Environment: Received unexpected action format for TLS {tls_id}: {phase_index}. Expected phase index (int).")

        # Run simulation steps AFTER attempting to apply all actions for this control cycle
        # Only step if at least one action was successfully applied (or attempted)
        if applied_actions_count > 0 or True: # Step even if no action applied to advance time
            for _ in range(self.network.simulation_steps_per_action):
                try:
                    self.network.simulation_step()
                    # Update current time after each simulation step
                    self.current_time = self.network.get_current_time()
                except Exception as e:
                     self.logger.critical(f"Environment: Error during simulation step: {e}. Simulation likely disconnected.", exc_info=True)
                     # Stop stepping if simulation has issues
                     break

        # Update network state (like arrived vehicles) after stepping
        try:
            # self.network.update_edge_data() # May not be needed if get_state reads fresh metrics
            self.network.update_arrivals()
        except Exception as e:
            self.logger.error(f"Environment: Error during post-step update: {e}", exc_info=True)


    def is_terminal_state(self) -> bool:
        """Check if the current state is terminal (simulation ended).

        Returns:
            True if the simulation is complete, False otherwise
        """
        # Use the network interface to check terminal state
        return self.network.is_simulation_complete()


    # --- Deprecated Method ---
    # This method was specific to link-level control and is not needed for phase control.
    # def get_time_since_last_change(self, tls_id: str, link_index: int) -> float:
    #     """Get the time since the last state change for a specific link."""
    #     # ... (Implementation removed as it's less relevant for phase control) ...
    #     pass

