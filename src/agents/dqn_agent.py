import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import json
import logging
from collections import deque
from typing import Dict, Any, Tuple, Optional, List, Union
import traci

import torch.nn.functional as F

from .agent import Agent

# Configure logger for this module
dqn_logger = logging.getLogger(__name__) # Use __name__ for module-level logger


# --- Simplified DQN Network (Phase-Based) ---
class PhaseDQN(nn.Module):
    """A simple MLP DQN that predicts Q-values for each phase."""
    def __init__(self, input_dim: int, num_phases: int):
        """
        Args:
            input_dim: Size of the flattened state vector.
            num_phases: Number of possible phases (actions).
        """
        super(PhaseDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), # Increased layer size
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_phases) # Output Q-value for each phase
        )
        dqn_logger.info(f"PhaseDQN initialized with input_dim={input_dim}, num_phases={num_phases}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Flattened state tensor, shape [batch_size, input_dim].
        Returns:
            q_values: Tensor of Q-values for each phase, shape [batch_size, num_phases].
        """
        return self.network(x)

# --- DQNAgent Modified for Phase Control ---
class DQNAgent(Agent):
    """Agent using Deep Q-Learning for traffic signal control (Phase-Based Actions)."""

    DEFAULT_CONFIG = {
        "alpha": 0.001,           # Learning rate (adjust as needed)
        "gamma": 0.95,            # Discount factor
        "epsilon": 1.0,           # Start with full exploration
        "epsilon_decay": 0.999,   # Slower decay
        "epsilon_min": 0.05,      # Lower minimum exploitation
        "batch_size": 64,
        "memory_size": 20000,     # Increased memory size
        "target_update_freq": 100 # Update target network more often
    }

    def __init__(self, tls_id, network, alpha=0.001, gamma=0.95, epsilon=1.0,
                 epsilon_decay=0.999, epsilon_min=0.05, batch_size=64,
                 memory_size=20000, target_update_freq=100):
        """Initialize DQNAgent for phase control."""
        super().__init__(tls_id, network)
        self.network = network # Keep network reference

        # DQN hyperparameters from config
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.target_update_freq = target_update_freq

        # --- Action Space: Phases ---
        try:
            # Get phase definitions from SUMO logic for this TLS
            logics = traci.trafficlight.getAllProgramLogics(self.tls_id)
            if not logics:
                raise ValueError(f"No program logics found for TLS '{self.tls_id}'")
            # Assuming the first logic is the relevant one
            self.phases = logics[0].phases
            self.num_phases = len(self.phases)
            self.action_size = self.num_phases # Action is selecting a phase index
            dqn_logger.info(f"TLS {self.tls_id} initialized with {self.num_phases} phases.")
        except Exception as e:
             dqn_logger.error(f"Failed to get phases for TLS {self.tls_id}: {e}. Setting num_phases to 4 as fallback.")
             # Fallback if TraCI fails or no logic defined
             self.num_phases = 4 # Common fallback
             self.action_size = 4
             self.phases = None # Indicate phases couldn't be loaded

        # --- State Representation ---
        # Simplified state: Features derived from links, plus current phase info
        # Example features: max_queue, total_queue, avg_wait, time_on_current_phase, current_phase_one_hot
        self.link_feature_dim = 4 # queue, wait, count, time_since_change
        self.max_links = 16 # Max links to consider (adjust based on networks)
        # Flattened link features + one-hot encoded phase
        self.state_dim = (self.max_links * self.link_feature_dim) + self.num_phases
        dqn_logger.info(f"State dimension set to {self.state_dim}")

        # Experience replay memory
        self.memory = deque(maxlen=self.memory_size)

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dqn_logger.info(f"DQNAgent {self.tls_id} using device: {self.device}")

        # Create main and target networks (using PhaseDQN)
        self.model = PhaseDQN(self.state_dim, self.action_size).to(self.device)
        self.target_model = PhaseDQN(self.state_dim, self.action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

        # Tracking
        self.train_step = 0
        self.last_action = None # Stores the chosen phase index
        self.current_phase_start_time = 0.0 # Track time for state feature

    @classmethod
    def create(cls, tls_id, network, **kwargs):
        """Factory method for DQNAgent."""
        config = cls.DEFAULT_CONFIG.copy()
        config.update(kwargs)
        return cls(tls_id, network, **config)

    def _preprocess_state(self, state: Dict[str, Any]) -> Optional[np.ndarray]:
        """Convert environment state dict to a flattened numpy array for the network."""
        if not isinstance(state, dict) or 'link_states' not in state or 'current_signal_state' not in state:
            dqn_logger.warning(f"Invalid state format received for {self.tls_id}")
            return None

        link_states = state.get('link_states', [])
        # current_signal_state = state.get('current_signal_state', "") # Not directly used now
        current_time = self.network.get_current_time() # Get current time from network

        # --- Calculate Current Phase ---
        # We need to know the current phase index to create the one-hot encoding
        # and calculate time_on_phase. This might require querying SUMO again.
        try:
            current_phase_index = traci.trafficlight.getPhase(self.tls_id)
        except Exception as e:
             dqn_logger.warning(f"Could not get current phase index for {self.tls_id}: {e}. Defaulting to 0.")
             current_phase_index = 0 # Fallback

        # Check if phase changed to reset timer
        if self.last_action != current_phase_index:
             self.current_phase_start_time = current_time

        time_on_phase = current_time - self.current_phase_start_time

        # --- Extract Link Features ---
        link_features = []
        for link in link_states:
            # Basic normalization (consider improving this)
            features = [
                link.get('queue_length', 0.0) / 20.0,
                link.get('waiting_time', 0.0) / 300.0,
                link.get('vehicle_count', 0.0) / 30.0,
                link.get('time_since_last_change', 0.0) / 120.0
            ]
            link_features.append(features)

        # Pad or truncate link features
        num_actual_links = len(link_features)
        if num_actual_links < self.max_links:
            padding = [[0.0] * self.link_feature_dim] * (self.max_links - num_actual_links)
            link_features.extend(padding)
        elif num_actual_links > self.max_links:
            link_features = link_features[:self.max_links]

        # Flatten link features
        flat_link_features = np.array(link_features).flatten()

        # --- Create Phase One-Hot Encoding ---
        phase_one_hot = np.zeros(self.num_phases)
        if 0 <= current_phase_index < self.num_phases:
             phase_one_hot[current_phase_index] = 1.0
        else:
            dqn_logger.warning(f"Current phase index {current_phase_index} out of bounds for one-hot encoding (num_phases={self.num_phases}).")
            # Handle out-of-bounds index, e.g., set the first element or leave as zeros
            if self.num_phases > 0: phase_one_hot[0] = 1.0 # Default to phase 0

        # --- Combine Features ---
        # Add normalized time_on_phase to the state
        norm_time_on_phase = time_on_phase / 60.0 # Normalize by estimated max phase time

        # Final state vector
        state_vector = np.concatenate((
            flat_link_features,
            phase_one_hot
            # Potentially add norm_time_on_phase here if desired
            # np.array([norm_time_on_phase])
        )).astype(np.float32)

        # Ensure state_vector dimension matches self.state_dim (adjust self.state_dim if needed)
        if len(state_vector) != self.state_dim:
             dqn_logger.error(f"State vector length mismatch! Expected {self.state_dim}, Got {len(state_vector)}. Check feature calculation.")
             # Fallback: return None or a zero vector of correct size
             return np.zeros(self.state_dim, dtype=np.float32) # Return zeros to avoid crashing downstream

        return state_vector

    def choose_action(self, state: Dict[str, Any]) -> Optional[int]:
        """Choose a phase index based on the current state."""
        processed_state = self._preprocess_state(state)

        if processed_state is None:
            return None # Cannot choose action

        action_index = None
        # Exploration vs Exploitation
        if np.random.rand() <= self.epsilon:
            # --- Explore: Choose a random phase index ---
            action_index = random.randrange(self.action_size)
        else:
            # --- Exploit: Choose the best phase based on Q-values ---
            state_tensor = torch.FloatTensor(processed_state).unsqueeze(0).to(self.device)
            self.model.eval() # Evaluation mode
            with torch.no_grad():
                q_values = self.model(state_tensor) # Shape: [1, num_phases]
            self.model.train() # Training mode

            # --- Action Masking (Optional but Recommended) ---
            # Check if the selected action corresponds to the current phase.
            # Sometimes staying in the current phase is needed (green extension).
            # No inherent conflict check needed when selecting phases, SUMO handles it.
            # However, you *could* mask actions that would violate min_green times if you track them.
            # For now, we allow selecting any phase.
            action_index = torch.argmax(q_values).item()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.last_action = action_index # Store chosen phase index

        return action_index # Return the phase index

    def remember(self, state_dict, action_index, reward, next_state_dict, done):
        """Store experience in replay memory."""
        if action_index is None: return # Can't store if no action taken

        state_vector = self._preprocess_state(state_dict)
        next_state_vector = self._preprocess_state(next_state_dict)

        if state_vector is None or next_state_vector is None:
            # dqn_logger.debug("Skipping remember due to invalid state preprocessing.")
            return # Don't store incomplete transitions

        # Store: state_vector, action_index, reward, next_state_vector, done
        experience = (state_vector, action_index, reward, next_state_vector, done)
        self.memory.append(experience)

    def learn(self, state: Dict[str, Any], action_index: Optional[int], next_state: Dict[str, Any], done: bool):
        """Learn from experience using DQN."""
        if action_index is None: return # Can't learn without an action

        # Calculate reward (using the base class method for now)
        # The base reward function might need the original action tuple format.
        # Let's adapt or use a simplified reward for phase control.
        reward, _ = self.calculate_reward_phase_based(state, action_index, next_state)

        # Store experience
        self.remember(state, action_index, reward, next_state, done)

        # Perform experience replay if memory is sufficient
        if len(self.memory) >= self.batch_size:
            self._replay()

    def _replay(self):
        """Perform experience replay."""
        minibatch = random.sample(self.memory, self.batch_size)

        # Unpack batch data
        state_vectors = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        action_indices = torch.LongTensor([t[1] for t in minibatch]).unsqueeze(1).to(self.device) # Shape: [batch, 1]
        rewards = torch.FloatTensor([t[2] for t in minibatch]).to(self.device)
        next_state_vectors = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.device)
        dones = torch.FloatTensor([t[4] for t in minibatch]).to(self.device)

        # --- Q-value Calculation ---
        # 1. Get Q-values for actions taken in the original states
        # model output shape: [batch, num_phases]
        current_q_all = self.model(state_vectors)
        # Gather Q-values for the specific action indices
        current_q = torch.gather(current_q_all, 1, action_indices).squeeze(1) # Shape: [batch]

        # 2. Calculate target Q-values using Double DQN
        with torch.no_grad():
            # Select best actions for next states using the online model
            next_q_all_online = self.model(next_state_vectors) # Shape: [batch, num_phases]
            best_next_actions = torch.argmax(next_q_all_online, dim=1).unsqueeze(1) # Shape: [batch, 1]

            # Evaluate these actions using the target model
            next_q_all_target = self.target_model(next_state_vectors) # Shape: [batch, num_phases]
            next_q_target = torch.gather(next_q_all_target, 1, best_next_actions).squeeze(1) # Shape: [batch]

            # Calculate final target Q value
            target_q = rewards + self.gamma * next_q_target * (1 - dones)

        # 3. Calculate loss
        loss = F.mse_loss(current_q, target_q)

        # 4. Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # Clip gradients
        self.optimizer.step()

        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def calculate_reward_phase_based(self, state, action_index, next_state):
        """Calculate reward based on state changes for phase-based control."""
        if not isinstance(state, dict) or 'link_states' not in state or \
           not isinstance(next_state, dict) or 'link_states' not in next_state:
            return 0.0, {}

        prev_link_states = state.get('link_states', [])
        next_link_states = next_state.get('link_states', [])

        if not prev_link_states or not next_link_states:
            return 0.0, {}

        # Simple reward: Negative change in total waiting time
        prev_total_wait = sum(link.get('waiting_time', 0) for link in prev_link_states)
        next_total_wait = sum(link.get('waiting_time', 0) for link in next_link_states)

        # Reward is the reduction in waiting time
        wait_reward = (prev_total_wait - next_total_wait) / 10.0 # Scale down reward

        # Calculate intersection-specific throughput
        # Count vehicles that have passed through the intersection
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
        
        # Reward for intersection throughput
        throughput_reward = passed_vehicles * 0.2  # Scale for appropriate impact
        
        # Small penalty for very long queues to discourage gridlock
        max_queue = 0
        for link in next_link_states:
            max_queue = max(max_queue, link.get('queue_length', 0))

        queue_penalty = 0
        if max_queue > 15: # Penalize if any queue gets very long
            queue_penalty = - (max_queue - 15) * 0.05

        # Combine rewards
        total_reward = wait_reward + throughput_reward + queue_penalty

        components = {
            'wait_time_reduction': wait_reward,
            'throughput_reward': throughput_reward,
            'queue_penalty': queue_penalty,
            'passed_vehicles': passed_vehicles
        }

        return total_reward, components

    # --- save_state and load_state (Keep the existing ones, they should work) ---
    def save_state(self, directory_path: str):
        """Saves the DQN agent's state."""
        # (Existing save_state logic - check it matches attribute names like self.num_phases)
        try:
            if hasattr(super(), 'save_state') and callable(super().save_state):
                 super().save_state(directory_path)
            else:
                 os.makedirs(directory_path, exist_ok=True)
        except Exception as e:
            dqn_logger.error(f"Error ensuring directory exists {directory_path}: {e}", exc_info=True)
            return

        model_path = os.path.join(directory_path, 'model.pth')
        target_model_path = os.path.join(directory_path, 'target_model.pth')
        optimizer_path = os.path.join(directory_path, 'optimizer.pth')
        hyperparams_path = os.path.join(directory_path, 'hyperparams.json')
        dqn_logger.info(f"Attempting to save Phase DQNAgent state for {self.tls_id} to {directory_path}")

        try:
            torch.save(self.model.state_dict(), model_path)
            torch.save(self.target_model.state_dict(), target_model_path)
            torch.save(self.optimizer.state_dict(), optimizer_path)

            hyperparams = {
                'epsilon': self.epsilon,
                'gamma': self.gamma,
                'alpha': self.alpha, # Save actual learning rate used
                'train_step': self.train_step,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
                'batch_size': self.batch_size,
                'memory_size': self.memory_size,
                'target_update_freq': self.target_update_freq,
                'state_dim': self.state_dim, # Save state dim used by network
                'action_size': self.action_size, # Save action size (num_phases)
                'num_phases': self.num_phases,
                'max_links': self.max_links,
                'link_feature_dim': self.link_feature_dim
            }
            with open(hyperparams_path, 'w') as f:
                json.dump(hyperparams, f, indent=4)

            dqn_logger.info(f"Phase DQNAgent state for {self.tls_id} saved successfully.")

        except Exception as e:
            dqn_logger.error(f"Error saving Phase DQNAgent state for {self.tls_id}: {e}", exc_info=True)

    def load_state(self, directory_path: str):
        """Loads the DQN agent's state."""
        # (Existing load_state logic - check it correctly re-initializes models if needed)
        model_path = os.path.join(directory_path, 'model.pth')
        target_model_path = os.path.join(directory_path, 'target_model.pth')
        optimizer_path = os.path.join(directory_path, 'optimizer.pth')
        hyperparams_path = os.path.join(directory_path, 'hyperparams.json')
        dqn_logger.info(f"Attempting to load Phase DQNAgent state for {self.tls_id} from {directory_path}")

        required_files = [model_path, target_model_path, hyperparams_path] # Optimizer optional
        if not all(os.path.exists(p) for p in required_files):
            dqn_logger.warning(f"Cannot load Phase DQNAgent state: Required file(s) not found in {directory_path}")
            return

        try:
            # --- Load Hyperparameters First (to potentially re-init models/optimizer) ---
            loaded_state_dim = None
            loaded_action_size = None
            loaded_alpha = self.alpha # Default to current
            with open(hyperparams_path, 'r') as f:
                hyperparams = json.load(f)
                self.epsilon = hyperparams.get('epsilon', self.epsilon)
                self.gamma = hyperparams.get('gamma', self.gamma)
                self.train_step = hyperparams.get('train_step', self.train_step)
                self.epsilon_decay = hyperparams.get('epsilon_decay', self.epsilon_decay)
                self.epsilon_min = hyperparams.get('epsilon_min', self.epsilon_min)
                loaded_alpha = hyperparams.get('alpha', self.alpha)
                loaded_state_dim = hyperparams.get('state_dim')
                loaded_action_size = hyperparams.get('action_size')
                # Load other params like batch_size, target_update_freq if they might change

            # --- Validate and Potentially Re-initialize Models ---
            if loaded_state_dim != self.state_dim or loaded_action_size != self.action_size:
                 dqn_logger.warning(f"Saved model dimensions ({loaded_state_dim}x{loaded_action_size}) mismatch current ({self.state_dim}x{self.action_size}). Re-initializing models.")
                 self.model = PhaseDQN(self.state_dim, self.action_size).to(self.device)
                 self.target_model = PhaseDQN(self.state_dim, self.action_size).to(self.device)
            # Now load state dicts
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.target_model.load_state_dict(torch.load(target_model_path, map_location=self.device))

            # --- Load Optimizer ---
            # Re-initialize optimizer with loaded alpha BEFORE loading state dict
            self.optimizer = optim.Adam(self.model.parameters(), lr=loaded_alpha)
            if os.path.exists(optimizer_path):
                self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            else:
                dqn_logger.warning("Optimizer state file not found. Optimizer initialized with loaded alpha.")

            self.model.to(self.device)
            self.target_model.to(self.device)
            self.target_model.eval()

            dqn_logger.info(f"Phase DQNAgent state loaded successfully for {self.tls_id}. Epsilon={self.epsilon:.4f}, TrainStep={self.train_step}")

        except Exception as e:
            dqn_logger.error(f"Error loading Phase DQNAgent state for {self.tls_id}: {e}", exc_info=True)
