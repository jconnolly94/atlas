import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import json
import pickle
import logging
from collections import deque, defaultdict
from typing import Dict, Any, Tuple, Optional, List, Union

from .agent import Agent

# Configure logger for this module
adv_logger = logging.getLogger(__name__) # Use __name__ for module-level logger
# Basic config if run standalone, or rely on project's main config
if not adv_logger.hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class AdvancedLinkEncoder(nn.Module):
    """Advanced encoder for link-level state representation with attention mechanism."""

    def __init__(self, input_dim=6, embedding_dim=16, dropout_rate=0.1):
        """Initialize link encoder.

        Args:
            input_dim: Dimensionality of each link state vector
            embedding_dim: Dimensionality of output embeddings
            dropout_rate: Dropout rate for regularization
        """
        super(AdvancedLinkEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, embedding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """Encode link state vectors.

        Args:
            x: Tensor of shape [batch_size, num_links, input_dim]

        Returns:
            Tensor of shape [batch_size, num_links, embedding_dim]
        """
        # Process each link independently
        batch_size, num_links, input_dim = x.shape
        x = x.view(-1, input_dim)  # Reshape to [batch_size * num_links, input_dim]
        x = self.encoder(x)
        return x.view(batch_size, num_links, -1)  # Reshape back to [batch_size, num_links, embedding_dim]


class AdvancedSignalStateEncoder(nn.Module):
    """Advanced encoder for signal state with attention mechanism."""

    def __init__(self, embedding_dim=16):
        """Initialize signal state encoder.

        Args:
            embedding_dim: Dimensionality of output embeddings
        """
        super(AdvancedSignalStateEncoder, self).__init__()
        # Character-level embedding for signal state (r, R, g, G, y, Y, o, O)
        self.char_embedding = nn.Embedding(8, embedding_dim)

        # Attention mechanism for signal state
        self.attention = nn.Linear(embedding_dim, 1)

        # Mapping from signal state characters to indices
        self.char_to_idx = {
            'r': 0, 'R': 1, 'g': 2, 'G': 3,
            'y': 4, 'Y': 5, 'o': 6, 'O': 7
        }

    def _state_to_indices(self, state_str):
        """Convert signal state string to tensor of indices.

        Args:
            state_str: Signal state string (e.g., "GrGr")

        Returns:
            Tensor of indices
        """
        indices = []
        for c in state_str:
            if c in self.char_to_idx:
                indices.append(self.char_to_idx[c])
            else:
                # Default to 'r' for unknown characters
                indices.append(0)
        return torch.tensor(indices, dtype=torch.long)

    def forward(self, state_strs):
        """Encode signal state strings with attention mechanism.

        Args:
            state_strs: List of signal state strings

        Returns:
            Tensor of attended embeddings for each position in the state strings
        """
        # Process batch of state strings
        batch_embeddings = []

        # Add device handling for indices tensor
        device = next(self.parameters()).device

        for state_str in state_strs:
            indices = self._state_to_indices(state_str).to(device) # Move indices to correct device
            char_embeddings = self.char_embedding(indices)  # [seq_len, embedding_dim]

            # Apply attention
            attention_scores = self.attention(char_embeddings)  # [seq_len, 1]
            attention_weights = F.softmax(attention_scores, dim=0)  # [seq_len, 1]

            # Weight character embeddings by attention
            weighted_embeddings = char_embeddings * attention_weights  # [seq_len, embedding_dim]
            batch_embeddings.append(weighted_embeddings)

        try:
             # Simple stacking assuming equal length for now
             return torch.stack(batch_embeddings)  # [batch_size, seq_len, embedding_dim]
        except RuntimeError as e:
             adv_logger.error(f"Error stacking signal embeddings (lengths might differ?): {e}")
             return torch.zeros(0)


class AdvancedPhaseDQN(nn.Module):
    """Advanced Deep Q-Network for traffic signal control with phase-based architecture.
    
    This is a phase-based model that outputs Q-values for each traffic light phase,
    similar to the DQN agent but with a more sophisticated architecture.
    """

    def __init__(self, max_links=16, link_dim=8, num_phases=4):
        """Initialize the phase-based network architecture.

        Args:
            max_links: Maximum number of links to support
            link_dim: Dimensionality of each link's feature vector
            num_phases: Number of possible phases (actions)
        """
        super(AdvancedPhaseDQN, self).__init__()

        # Embedding dimensions
        self.link_embedding_dim = 16
        self.max_links = max_links
        self.num_phases = num_phases

        # Link encoder - transforms link features into embeddings
        self.link_encoder = AdvancedLinkEncoder(link_dim, self.link_embedding_dim)
        
        # Feature aggregation layer (combines all link embeddings into a single state representation)
        self.feature_aggregator = nn.Sequential(
            nn.Linear(self.link_embedding_dim * max_links, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Phase selection network (outputs Q-values for each phase)
        self.phase_selector = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_phases)  # Q-values for each phase
        )

        # Historical trend analyzer for long-term patterns (for enhanced state representation)
        self.trend_analyzer = nn.Sequential(
            nn.Linear(self.link_embedding_dim * max_links, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)  # Trend features
        )

    def forward(self, link_features, signal_state=None):
        """Forward pass through the network with phase-based architecture.

        Args:
            link_features: Tensor of link features [batch_size, num_links, link_dim]
            signal_state: (Optional) List of signal state strings (not directly used in phase-based model)

        Returns:
            phase_q_values: Q-values for each phase [batch_size, num_phases]
            trend_features: Features representing traffic trends [batch_size, 8]
        """
        batch_size = link_features.shape[0]
        num_links = link_features.shape[1]
        device = link_features.device
        
        # Ensure we have at least one sample to process
        if batch_size == 0 or num_links == 0:
            # Return empty tensors of appropriate shapes if no data
            return (torch.zeros(0, self.num_phases, device=device),
                    torch.zeros(0, 8, device=device))  # 8 trend features

        # Encode link features
        link_embeddings = self.link_encoder(link_features)  # [batch_size, num_links, link_embedding_dim]
        
        # Flatten link embeddings for each sample in the batch
        flat_embeddings = link_embeddings.reshape(batch_size, -1)  # [batch_size, num_links * link_embedding_dim]
        
        # Aggregate features
        aggregated_features = self.feature_aggregator(flat_embeddings)  # [batch_size, 128]
        
        # Get Q-values for each phase
        phase_q_values = self.phase_selector(aggregated_features)  # [batch_size, num_phases]
        
        # Extract trend features for enhanced state representation
        trend_features = self.trend_analyzer(flat_embeddings)  # [batch_size, 8]
        
        return phase_q_values, trend_features


class AdvancedAgent(Agent):
    """Advanced agent using Deep Q-Learning with sophisticated architecture for lane-level traffic control."""

    # Default configuration - Updated with DQN agent learnings
    DEFAULT_CONFIG = {
        "alpha": 0.001,           # Matched to DQN agent's learning rate
        "gamma": 0.95,            # Matched to DQN agent's discount factor
        "epsilon": 1.0,           # Start with full exploration like DQN
        "epsilon_decay": 0.999,   # Slower decay like DQN
        "epsilon_min": 0.05,      # Lower minimum exploitation like DQN
        "batch_size": 64,         # Keep batch size
        "memory_size": 20000,     # Reduced to DQN's size (proven to be sufficient)
        "target_update_freq": 100, # Update target network more often like DQN
        "double_dqn": True,       # Keep double DQN for more stable learning
        "prioritized_replay": True, # Keep prioritized experience replay
        "priority_alpha": 0.6,    # Keep priority exponent for PER
        "priority_beta": 0.4      # Keep importance sampling exponent for PER
    }

    def __init__(self, tls_id, network, conflict_detector=None, alpha=0.001, gamma=0.95, epsilon=1.0,
                 epsilon_decay=0.999, epsilon_min=0.05, batch_size=64,
                 memory_size=20000, target_update_freq=100, double_dqn=True,
                 prioritized_replay=True, priority_alpha=0.6, priority_beta=0.4):
        """Initialize AdvancedAgent with phase-based control capabilities.
        Args:
            tls_id: Traffic light system ID
            network: Network interface
            conflict_detector: (Optional, not used in phase-based control)
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Exploration decay rate
            epsilon_min: Minimum exploration rate
            batch_size: Training batch size
            memory_size: Replay memory size
            target_update_freq: Target network update frequency
            double_dqn: Whether to use double DQN
            prioritized_replay: Whether to use prioritized experience replay
            priority_alpha: Exponent alpha for prioritized experience replay
            priority_beta: Exponent beta for prioritized experience replay importance sampling
        """
        super().__init__(tls_id, network)
        self.network = network
        # conflict_detector not needed for phase-based control
        
        # --- Initialize Phase Information (from DQN) ---
        try:
            # Get phase definitions from network interface
            self.phases = self.network.get_traffic_light_phases(self.tls_id)
            if not self.phases:
                raise ValueError(f"No program logics found for TLS '{self.tls_id}'")
            self.num_phases = len(self.phases)
            self.action_size = self.num_phases  # Action is selecting a phase index
            adv_logger.info(f"TLS {self.tls_id} initialized with {self.num_phases} phases.")
        except Exception as e:
            adv_logger.error(f"Failed to get phases for TLS {self.tls_id}: {e}. Setting num_phases to 4 as fallback.")
            # Fallback if network interface fails
            self.num_phases = 4  # Common fallback
            self.action_size = 4
            self.phases = None  # Indicate phases couldn't be loaded
        
        # Maximum number of links and features per link
        self.max_links = 26  # Support more links than standard DQN
        self.link_dim = 8    # Expanded with phase information and time on phase

        # DQN hyperparameters from config
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.prioritized_replay = prioritized_replay
        self.priority_alpha = priority_alpha
        self.priority_beta = priority_beta
        self.priority_beta_increment = 0.001
        self.priority_epsilon = 0.01  # Small constant to avoid zero priority
        
        # Experience replay memory with prioritization if enabled
        if self.prioritized_replay:
            # Prioritized replay uses a list of (experience, priority) tuples
            self.memory = []
        else:
            # Standard replay memory
            self.memory = deque(maxlen=self.memory_size)

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adv_logger.info(f"AdvancedAgent {self.tls_id} using device: {self.device}")

        # Create main and target networks with advanced phase-based architecture
        self.model = AdvancedPhaseDQN(self.max_links, self.link_dim, self.num_phases).to(self.device)
        self.target_model = AdvancedPhaseDQN(self.max_links, self.link_dim, self.num_phases).to(self.device)

        # Copy weights from main to target network
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # Target network should be in eval mode

        # Optimizer with weight decay for regularization
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha, weight_decay=1e-5)

        # Target network update frequency
        self.train_step = 0

        # Track last action (phase index) and state for reward calculation
        self.last_action = None  # Now stores the chosen phase index
        self.last_state = None
        self.last_phase_index = 0
        
        # Traffic pattern history for advanced analysis
        self.traffic_history = defaultdict(lambda: deque(maxlen=50))  # Store last 50 observations per link

        # Performance tracking for adaptive exploration
        self.performance_history = deque(maxlen=20)  # Store last 20 rewards
        
        # Initialize running statistics for online normalization
        self.stats_count = torch.tensor(0, device=self.device)
        # Create tensors for each feature type: [queue_length, waiting_time, vehicle_count, time_since_last_change]
        self.feature_means = torch.zeros(4, device=self.device)
        self.feature_variances = torch.zeros(4, device=self.device)
        
        # Track phase-related information
        self.current_phase_start_time = 0.0  # Track time on current phase

    # Methods for link-level control have been removed in favor of phase-based control
    # No need for conflict detection or action masking in phase-based control
    # as SUMO handles phase conflicts internally

    @classmethod
    def create(cls, tls_id, network, **kwargs):
        """Create an instance of the AdvancedAgent with proper configuration.
        # ... (rest of docstring) ...
        """
        # Start with default configuration
        config = cls.DEFAULT_CONFIG.copy()

        # Override with provided kwargs
        config.update(kwargs)

        # Create and return instance
        return cls(tls_id, network, **config)

    def _preprocess_state(self, state):
        """Convert environment state dict to tensors for the neural network.
        Applies online standardization to numerical features for improved learning stability.
        Now also includes phase information from DQN agent approach.
        """
        # Extract link states and current signal state
        link_states = state.get('link_states', []) # Use .get for safety
        current_signal_state = state.get('current_signal_state', "")
        
        # --- Get Phase Information (from DQN approach) ---
        try:
            # Use the network interface to get current phase index
            current_phase_index = self.network.get_current_phase_index(self.tls_id)
            current_time = self.network.get_current_time()
            
            # If phase changed, reset the phase timer
            if hasattr(self, 'last_phase_index') and self.last_phase_index != current_phase_index:
                self.current_phase_start_time = current_time
                
            # Calculate time on current phase
            time_on_phase = current_time - self.current_phase_start_time
            # Normalize time on phase
            norm_time_on_phase = min(1.0, time_on_phase / 60.0)  # Normalize to [0,1]
            
            # Store current phase for next comparison
            self.last_phase_index = current_phase_index
            
        except Exception as e:
            # If getting phase info fails, use defaults
            adv_logger.warning(f"Could not get phase info in preprocess: {e}")
            current_phase_index = 0
            norm_time_on_phase = 0.0

        # Collect all raw feature values first to update statistics
        all_queue_lengths = []
        all_waiting_times = []
        all_vehicle_counts = []
        all_time_since_changes = []

        # First pass: collect features for statistics update
        for link in link_states:
            link_index = link.get('index', -1)
            if link_index == -1 or link_index >= len(current_signal_state):
                continue # Skip if index is invalid or out of bounds
                
            queue_length = float(link.get('queue_length', 0))
            waiting_time = float(link.get('waiting_time', 0))
            vehicle_count = float(link.get('vehicle_count', 0))
            time_since_change = float(link.get('time_since_last_change', 0))
            
            all_queue_lengths.append(queue_length)
            all_waiting_times.append(waiting_time)
            all_vehicle_counts.append(vehicle_count)
            all_time_since_changes.append(time_since_change)

        # Update running statistics if we have data
        if all_queue_lengths:  # Only update if we have data
            # Convert collected data to tensors
            queue_tensor = torch.tensor(all_queue_lengths, dtype=torch.float32, device=self.device)
            waiting_tensor = torch.tensor(all_waiting_times, dtype=torch.float32, device=self.device)
            vehicles_tensor = torch.tensor(all_vehicle_counts, dtype=torch.float32, device=self.device)
            time_change_tensor = torch.tensor(all_time_since_changes, dtype=torch.float32, device=self.device)

            # Update running statistics using Welford's algorithm
            self.stats_count += 1
            
            # Queue length (index 0)
            delta = queue_tensor.mean() - self.feature_means[0]
            self.feature_means[0] += delta / self.stats_count
            self.feature_variances[0] += delta * (queue_tensor.mean() - self.feature_means[0])
            
            # Waiting time (index 1)
            delta = waiting_tensor.mean() - self.feature_means[1]
            self.feature_means[1] += delta / self.stats_count
            self.feature_variances[1] += delta * (waiting_tensor.mean() - self.feature_means[1])
            
            # Vehicle count (index 2)
            delta = vehicles_tensor.mean() - self.feature_means[2]
            self.feature_means[2] += delta / self.stats_count
            self.feature_variances[2] += delta * (vehicles_tensor.mean() - self.feature_means[2])
            
            # Time since change (index 3)
            delta = time_change_tensor.mean() - self.feature_means[3]
            self.feature_means[3] += delta / self.stats_count
            self.feature_variances[3] += delta * (time_change_tensor.mean() - self.feature_means[3])

        # Second pass: process features with normalization and build final feature list
        link_features = []

        for link in link_states:
            link_index = link.get('index', -1)
            if link_index == -1 or link_index >= len(current_signal_state):
                continue # Skip if index is invalid or out of bounds

            # Check if current state is green (G or g)
            is_green = 1.0 if current_signal_state[link_index] in 'Gg' else 0.0

            # Calculate historical trend (average queue growth over recent history)
            historical_trend = 0.0
            if link_index in self.traffic_history and len(self.traffic_history[link_index]) > 1:
                queue_history = [record.get('queue_length', 0) for record in self.traffic_history[link_index]]
                if len(queue_history) >= 2:
                    # Calculate average rate of change
                    changes = [(queue_history[i] - queue_history[i-1]) for i in range(1, len(queue_history))]
                    if changes: # Avoid division by zero if only 2 elements and no change
                        historical_trend = sum(changes) / len(changes)
                        # Normalize to range [-1, 1]
                        historical_trend = max(-1.0, min(1.0, historical_trend / 5.0))

            # Get raw feature values
            queue_length = float(link.get('queue_length', 0))
            waiting_time = float(link.get('waiting_time', 0))
            vehicle_count = float(link.get('vehicle_count', 0))
            time_since_change = float(link.get('time_since_last_change', 0))
            
            # Apply normalization with epsilon for numerical stability
            # Only use standardization if we have enough samples
            if self.stats_count > 1:
                # Calculate standard deviation with small epsilon to avoid division by zero
                epsilon = 1e-8
                std_queue = torch.sqrt(self.feature_variances[0] / self.stats_count + epsilon)
                std_waiting = torch.sqrt(self.feature_variances[1] / self.stats_count + epsilon)
                std_vehicles = torch.sqrt(self.feature_variances[2] / self.stats_count + epsilon)
                std_time_change = torch.sqrt(self.feature_variances[3] / self.stats_count + epsilon)
                
                # Normalize features
                norm_queue = (queue_length - self.feature_means[0]) / std_queue
                norm_waiting = (waiting_time - self.feature_means[1]) / std_waiting
                norm_vehicles = (vehicle_count - self.feature_means[2]) / std_vehicles
                norm_time_change = (time_since_change - self.feature_means[3]) / std_time_change
                
                # Clip to recommended range [-5.0, 5.0]
                norm_queue = max(-5.0, min(5.0, norm_queue.item()))
                norm_waiting = max(-5.0, min(5.0, norm_waiting.item()))
                norm_vehicles = max(-5.0, min(5.0, norm_vehicles.item()))
                norm_time_change = max(-5.0, min(5.0, norm_time_change.item()))
            else:
                # Fallback to basic scaling for the first few samples
                norm_queue = queue_length / 30.0
                norm_waiting = waiting_time / 1000.0
                norm_vehicles = vehicle_count / 30.0
                norm_time_change = time_since_change / 200.0
            
            # --- Add phase-specific information (like DQN does) ---
            # Add signals indicating if this link belongs to current phase
            # This helps the agent understand which links are related to the current phase
            link_in_current_phase = 0.0
            if self.phases and 0 <= current_phase_index < len(self.phases):
                # Check if this link is green in the current phase
                phase_state = self.phases[current_phase_index].state
                if link_index < len(phase_state) and phase_state[link_index] in 'Gg':
                    link_in_current_phase = 1.0
            
            # Assemble the feature vector with phase information
            features = [
                norm_queue,
                norm_waiting,
                norm_vehicles,
                norm_time_change,
                is_green,
                historical_trend,
                norm_time_on_phase,     # Add time on current phase
                link_in_current_phase   # Add flag if link belongs to current phase
            ]
            link_features.append(features)

            # Update traffic history for this link
            self.traffic_history[link_index].append({
                'queue_length': queue_length,
                'waiting_time': waiting_time,
                'vehicle_count': vehicle_count
            })

        # Pad to max_links if necessary
        num_actual_links = len(link_features)
        while len(link_features) < self.max_links:
            link_features.append([0.0] * len(features))  # Match feature length

        # Truncate if too many links (shouldn't happen if max_links is large enough)
        link_features = link_features[:self.max_links]

        return np.array(link_features), current_signal_state

    def choose_action(self, state):
        """Choose a phase index action based on the current state with advanced exploration strategies.
        This method has been completely rewritten to use phase-based control instead of link-level control.
        
        Args:
            state: Dictionary containing state information
            
        Returns:
            Optional[int]: Phase index to activate, or None if no action can be taken
        """
        # Store state for later use
        self.last_state = state

        # Handle invalid state format
        if not isinstance(state, dict) or 'link_states' not in state:
            adv_logger.warning(f"Advanced agent {self.tls_id} received incompatible state format.")
            return None

        # Get link states and validate
        link_states = state.get('link_states', [])
        if not link_states:
            # adv_logger.debug(f"No link states for {self.tls_id}, cannot choose action.")
            return None  # No links to control

        # Preprocess state for neural network
        link_features_np, signal_state_str = self._preprocess_state(state)

        # Adaptive exploration strategy based on recent performance
        exploration_rate = self.epsilon  # Default
        if len(self.performance_history) >= 5:  # Require a minimum history
            # Calculate the average of recent rewards
            recent_rewards = list(self.performance_history)
            recent_rewards_subset = recent_rewards[-min(5, len(recent_rewards)):]
            if recent_rewards_subset:  # Avoid division by zero
                avg_reward = sum(recent_rewards_subset) / len(recent_rewards_subset)
                # If rewards are consistently poor, increase exploration
                if avg_reward < -0.5:  # Threshold might need tuning
                    exploration_rate = min(1.0, self.epsilon * 1.2)

        action_index = None  # Initialize action to None (fallback)

        # Exploration: choose random action with probability epsilon
        if np.random.rand() <= exploration_rate:
            # --- Explore: Choose a random phase index ---
            action_index = random.randrange(self.action_size)
            adv_logger.debug(f"Exploring with random phase index {action_index} for {self.tls_id}")
        else:
            # --- Exploit: Choose the best phase based on Q-values ---
            link_tensor = torch.FloatTensor(link_features_np).unsqueeze(0).to(self.device)

            self.model.eval()  # Set model to evaluation mode for inference
            with torch.no_grad():
                # Get Q-values for each phase from the model
                phase_q_values, _ = self.model(link_tensor)

            self.model.train()  # Set model back to training mode

            # Choose the phase with the highest Q-value
            action_index = torch.argmax(phase_q_values).item()
            adv_logger.debug(f"Exploiting with best phase index {action_index} for {self.tls_id}")

        # Update epsilon with adaptive decay logic
        if len(self.performance_history) >= 10:
            recent_rewards = list(self.performance_history)
            # Ensure indices are valid
            split_point = max(0, len(recent_rewards) - 5)
            recent_avg = sum(recent_rewards[split_point:]) / max(1, len(recent_rewards) - split_point)
            older_avg = sum(recent_rewards[:split_point]) / max(1, split_point)

            # Check if performance history is sufficient for comparison
            if len(recent_rewards) - split_point > 0 and split_point > 0:
                if recent_avg > older_avg:
                    effective_decay = self.epsilon_decay * 1.001  # Slower decay
                else:
                    effective_decay = self.epsilon_decay * 0.999  # Faster decay
                self.epsilon = max(self.epsilon_min, self.epsilon * effective_decay)
            else:  # Not enough distinct history for comparison, use standard decay
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        else:
            # Standard decay if not enough history
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Store last action for learning
        self.last_action = action_index
        return action_index

    def remember(self, state, action_index, reward, next_state, done):
        """Store experience in replay memory with prioritization for phase-based control.
        
        Args:
            state: Current state dictionary
            action_index: Phase index that was selected (int)
            reward: Reward received for the action
            next_state: Next state dictionary after taking action
            done: Whether the episode is done
        """
        if action_index is None or state is None or next_state is None:
            adv_logger.warning("Skipping remember due to None input.")
            return

        try:
            # Preprocess states (creates padded arrays of size max_links)
            link_features, signal_state = self._preprocess_state(state)
            next_link_features, next_signal_state = self._preprocess_state(next_state)

            # Create experience tuple
            experience = (
                link_features,      # Numpy array [max_links, link_dim]
                action_index,       # Phase index (int)
                float(reward),      # Ensure reward is float
                next_link_features, # Numpy array [max_links, link_dim]
                bool(done)          # Ensure done is bool
            )

            # --- Store experience based on replay type ---
            if self.prioritized_replay:
                max_priority = 1.0
                if self.memory:
                    try:
                        # Filter out potential invalid entries before finding max
                        valid_priorities = [p for item, p in self.memory if isinstance(item, tuple) and isinstance(p, (float, int))]
                        if valid_priorities: max_priority = max(valid_priorities)
                    except (ValueError, TypeError) as e:
                         adv_logger.warning(f"Could not determine max priority, defaulting to 1.0. Error: {e}")
                         max_priority = 1.0
                self.memory.append((experience, max_priority))
                # Simple FIFO removal if memory exceeds size
                if len(self.memory) > self.memory_size: self.memory.pop(0)
            else:
                # Standard experience replay (deque handles maxlen)
                self.memory.append(experience)

        except Exception as e:
            adv_logger.error(f"Error remembering experience: {e}", exc_info=True)


    def learn(self, state, action_index, next_state, done):
        """Learn from experience using advanced deep Q-learning with phase-based control.
        
        Args:
            state: Current state dictionary
            action_index: Phase index that was selected (int)
            next_state: Next state dictionary after taking action
            done: Whether the episode is done
        """
        if action_index is None:
            return

        # Calculate reward using agent's own method
        reward, _ = self.calculate_reward(state, action_index, next_state)

        # Store reward in performance history
        self.performance_history.append(reward)

        # Store experience in memory (handles preprocessing)
        self.remember(state, action_index, reward, next_state, done)

        # Determine required memory size based on replay type
        min_memory_for_replay = self.batch_size
        current_memory_size = len(self.memory)

        # Only learn if we have enough samples
        if current_memory_size < min_memory_for_replay:
            return

        # Perform experience replay
        self._replay()


    def _replay(self):
        """Perform experience replay with phase-based control."""
        # Sample from memory
        if self.prioritized_replay:
            if not self.memory: return # Should be caught by learn(), but defensive check

            # Extract priorities and experiences separately
            experiences = [item[0] for item in self.memory]
            priorities = np.array([item[1] for item in self.memory])

            # Apply priority exponent
            probs = priorities ** self.priority_alpha
            probs /= probs.sum() # Normalize

            # Sample indices based on probabilities
            indices = np.random.choice(len(self.memory), self.batch_size, p=probs, replace=False) # Sample without replacement

            # Get the batch of experiences and corresponding priorities
            batch = [experiences[idx] for idx in indices]
            
            # Calculate importance sampling weights
            total_samples = len(self.memory)
            weights = (total_samples * probs[indices]) ** (-self.priority_beta)
            weights /= weights.max() # Normalize for stability
            importance_weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device) # Add feature dim for broadcasting

            # Increment beta towards 1
            self.priority_beta = min(1.0, self.priority_beta + self.priority_beta_increment)
        else:
            # Standard uniform sampling
            if not self.memory: return
            batch = random.sample(self.memory, self.batch_size)
            importance_weights = torch.ones(self.batch_size, 1, device=self.device) # Weights are 1
            indices = None  # Not used for standard replay

        # --- Process Batch for Phase-based Control ---
        # Unpack the batch, converting numpy arrays to tensors on the correct device
        link_features_batch = torch.FloatTensor(np.array([t[0] for t in batch])).to(self.device)
        action_indices = torch.LongTensor([t[1] for t in batch]).to(self.device) # Phase indices
        rewards_batch = torch.FloatTensor([t[2] for t in batch]).to(self.device)
        next_link_features_batch = torch.FloatTensor(np.array([t[3] for t in batch])).to(self.device)
        dones_batch = torch.FloatTensor([t[4] for t in batch]).to(self.device)

        # --- Get Current Q-values for the Taken Actions ---
        # Get Q-values for all phases
        current_q_all_phases, _ = self.model(link_features_batch)
        
        # Select the Q-value for the action (phase) actually taken
        # Extract Q-values for specific action indices
        current_q = current_q_all_phases.gather(1, action_indices.unsqueeze(1)).squeeze(1)

        # --- Calculate Target Q-values with Double DQN ---
        with torch.no_grad():
            # 1. Select actions using the online model
            next_q_online, _ = self.model(next_link_features_batch)
            best_next_actions = torch.argmax(next_q_online, dim=1)
            
            # 2. Evaluate those actions using the target model
            next_q_target, _ = self.target_model(next_link_features_batch)
            
            # Extract Q-values for the selected actions
            q_target_next = next_q_target.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)
            
            # Calculate target Q-values: reward + gamma * next_q * (1-done)
            target_q = rewards_batch + (1 - dones_batch) * self.gamma * q_target_next

        # Compute loss (TD Error) element-wise
        td_errors = target_q - current_q # Shape: [batch_size]
        
        # Apply importance sampling weights for prioritized replay
        loss = (importance_weights * (td_errors ** 2)).mean() # Weighted MSE loss

        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # Clip gradients
        self.optimizer.step()

        # Update priorities for prioritized replay
        if self.prioritized_replay and indices is not None:
            # Calculate new priorities based on absolute TD error
            new_priorities = torch.abs(td_errors).detach().cpu().numpy() + self.priority_epsilon

            # Update priorities in memory list
            for i, idx in enumerate(indices):
                # Ensure index is valid
                if idx < len(self.memory):
                    # Replace tuple: (experience, new_priority)
                    self.memory[idx] = (self.memory[idx][0], new_priorities[i])
                else:
                    adv_logger.warning(f"Index {idx} out of bounds for memory during priority update.")

        # Update target network if needed
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            adv_logger.info(f"Updating target network at step {self.train_step}")
            self.target_model.load_state_dict(self.model.state_dict())

    def calculate_reward(self, state: Dict[str, Any], action_index: Optional[int],
                         next_state: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate reward based on phase-based control metrics.
        Considers waiting time reduction, throughput, and queue pressure.
        
        Args:
            state: Previous state dictionary.
            action_index: Phase index that was selected.
            next_state: Resulting state dictionary.
        Returns:
            Tuple[float, Dict[str, float]]: Calculated reward and components.
        """
        # --- Input Validation ---
        if not isinstance(next_state, dict) or not isinstance(state, dict):
            # Use logger associated with the instance if available, otherwise default
            logger_instance = getattr(self, 'adv_logger', logging.getLogger(__name__))
            logger_instance.warning(f"Invalid state or next_state for reward calc (TLS: {self.tls_id})")
            return 0.0, {}
            
        # Get link states from both state and next_state
        prev_link_states = state.get('link_states', [])
        next_link_states = next_state.get('link_states', [])
        
        if not prev_link_states or not next_link_states:
            # No links, pressure is effectively zero
            return 0.0, {'total_queue_length': 0.0, 'pressure': 0.0, 'reward': 0.0}
            
        # --- Waiting time reduction reward (like DQN) ---
        prev_total_wait = sum(link.get('waiting_time', 0) for link in prev_link_states)
        next_total_wait = sum(link.get('waiting_time', 0) for link in next_link_states)
        wait_reward = (prev_total_wait - next_total_wait) / 10.0  # Scale down reward
        
        # --- Calculate throughput reward (like DQN) ---
        # Count vehicles that have passed through the intersection
        prev_vehicle_count = sum(link.get('vehicle_count', 0) for link in prev_link_states)
        next_vehicle_count = sum(link.get('vehicle_count', 0) for link in next_link_states)
        
        # If vehicle count decreases, it means vehicles have exited the links controlled by this TLS
        passed_vehicles = max(0, prev_vehicle_count - next_vehicle_count)
        
        # If we have link exit counts directly, use those instead
        prev_exit_count = sum(link.get('exit_count', 0) for link in prev_link_states)
        next_exit_count = sum(link.get('exit_count', 0) for link in next_link_states)
        if next_exit_count > prev_exit_count:
            passed_vehicles = next_exit_count - prev_exit_count
        
        # Reward for intersection throughput
        throughput_reward = passed_vehicles * 0.2  # Scale for appropriate impact
        
        # --- Calculate pressure-based reward (original advanced agent approach) ---
        total_queue_length = sum(link.get('queue_length', 0.0) for link in next_link_states)
        pressure = total_queue_length  # Using queue length as pressure proxy
        PRESSURE_SCALE_FACTOR = 20.0  # Adjust based on observed queue sums
        pressure_reward = -pressure / PRESSURE_SCALE_FACTOR
        
        # --- Queue penalty to discourage gridlock (like DQN) ---
        max_queue = 0
        for link in next_link_states:
            max_queue = max(max_queue, link.get('queue_length', 0))
        
        queue_penalty = 0
        if max_queue > 15:  # Penalize if any queue gets very long
            queue_penalty = -(max_queue - 15) * 0.05
        
        # --- Phase transition penalty ---
        # Penalize frequent phase changes - only apply if we have prior state info
        phase_change_penalty = 0.0
        if hasattr(self, 'last_phase_index') and action_index is not None:
            # If we changed phase and the previous phase was active for a short time,
            # apply a small penalty to discourage rapid switching
            current_time = self.network.get_current_time()
            time_in_prev_phase = current_time - self.current_phase_start_time
            
            if self.last_phase_index != action_index and time_in_prev_phase < 10.0:  # 10 seconds threshold
                # Apply a small penalty that decreases as time_in_prev_phase increases
                phase_change_penalty = -max(0, (10.0 - time_in_prev_phase) * 0.05)
                adv_logger.debug(f"Applied phase change penalty: {phase_change_penalty} for switching after {time_in_prev_phase}s")
            
            # Update records if phase changed
            if self.last_phase_index != action_index:
                self.current_phase_start_time = current_time
                self.last_phase_index = action_index
        
        # --- Combine all reward components ---
        # Use a weighted combination of all reward components
        # The weights can be tuned based on performance
        total_reward = (
            wait_reward * 0.4 +        # 40% weight on wait time reduction
            throughput_reward * 0.4 +  # 40% weight on throughput
            pressure_reward * 0.15 +   # 15% weight on pressure reduction
            queue_penalty +            # Queue penalty always applies fully
            phase_change_penalty * 0.05  # 5% weight on phase change penalty
        )
        
        # Clip total reward to reasonable range
        total_reward = max(-10.0, min(total_reward, 10.0))
        
        # Return reward and components for analysis
        components = {
            'wait_reward': wait_reward,
            'throughput_reward': throughput_reward,
            'pressure_reward': pressure_reward,
            'queue_penalty': queue_penalty,
            'phase_change_penalty': phase_change_penalty,
            'total_queue_length': total_queue_length,
            'passed_vehicles': passed_vehicles,
            'total_reward': total_reward
        }
        
        return total_reward, components

    # --- save_state and load_state methods ---
    def save_state(self, directory_path: str):
        """Saves the AdvancedAgent's state to the specified directory."""
        super().save_state(directory_path) # Creates directory via base class call
        adv_logger.info(f"Saving AdvancedAgent (Phase-based) state for {self.tls_id} to {directory_path}")

        model_path = os.path.join(directory_path, 'model.pth')
        target_model_path = os.path.join(directory_path, 'target_model.pth')
        optimizer_path = os.path.join(directory_path, 'optimizer.pth')
        hyperparams_path = os.path.join(directory_path, 'hyperparams.json')
        normalization_path = os.path.join(directory_path, 'normalization.pth')

        try:
            # Save models and optimizer
            torch.save(self.model.state_dict(), model_path)
            torch.save(self.target_model.state_dict(), target_model_path)
            torch.save(self.optimizer.state_dict(), optimizer_path)
            
            # Save normalization statistics
            torch.save({
                'stats_count': self.stats_count,
                'feature_means': self.feature_means,
                'feature_variances': self.feature_variances
            }, normalization_path)

            # Save hyperparameters and other relevant state
            hyperparams = {
                'epsilon': self.epsilon,
                'gamma': self.gamma,
                'alpha': self.alpha, # Learning rate used by optimizer
                'train_step': self.train_step,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
                'batch_size': self.batch_size,
                'memory_size': self.memory_size, # Configured max size
                'target_update_freq': self.target_update_freq,
                'double_dqn': self.double_dqn,
                'prioritized_replay': self.prioritized_replay,
                'priority_alpha': self.priority_alpha,
                'priority_beta': self.priority_beta, # Save current beta
                'num_phases': self.num_phases,
                'action_size': self.action_size,
                'max_links': self.max_links,
                'link_dim': self.link_dim
            }
            with open(hyperparams_path, 'w') as f:
                json.dump(hyperparams, f, indent=4)

            adv_logger.info(f"AdvancedAgent (Phase-based) state saved successfully for {self.tls_id}.")

        except Exception as e:
            adv_logger.error(f"Error saving AdvancedAgent state for {self.tls_id}: {e}", exc_info=True)


    def load_state(self, directory_path: str):
        """Loads the AdvancedAgent's state from the specified directory."""
        adv_logger.info(f"Loading AdvancedAgent (Phase-based) state for {self.tls_id} from {directory_path}")

        model_path = os.path.join(directory_path, 'model.pth')
        target_model_path = os.path.join(directory_path, 'target_model.pth')
        optimizer_path = os.path.join(directory_path, 'optimizer.pth')
        hyperparams_path = os.path.join(directory_path, 'hyperparams.json')
        normalization_path = os.path.join(directory_path, 'normalization.pth')

        # Check if essential files exist
        required_files = [model_path, target_model_path, optimizer_path, hyperparams_path]
        if not all(os.path.exists(p) for p in required_files):
            adv_logger.warning(f"Cannot load AdvancedAgent state for {self.tls_id}: Required file(s) not found in {directory_path}. Starting fresh.")
            # We can't properly re-initialize here because we need the network reference
            return

        try:
            # Determine device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            adv_logger.info(f"Loading state onto device: {device}")

            # First load hyperparameters to check if model architecture needs to be rebuilt
            with open(hyperparams_path, 'r') as f:
                hyperparams = json.load(f)
                
                # Check if dimensions match current configuration
                loaded_num_phases = hyperparams.get('num_phases', self.num_phases)
                loaded_action_size = hyperparams.get('action_size', self.action_size)
                loaded_max_links = hyperparams.get('max_links', self.max_links)
                loaded_link_dim = hyperparams.get('link_dim', self.link_dim)
                
                # If dimensions don't match, rebuild models
                if (loaded_num_phases != self.num_phases or 
                    loaded_action_size != self.action_size or
                    loaded_max_links != self.max_links or
                    loaded_link_dim != self.link_dim):
                    
                    adv_logger.warning(f"Model dimensions mismatch. Rebuilding models with saved dimensions.")
                    
                    # Update current dimensions to match loaded ones
                    self.num_phases = loaded_num_phases
                    self.action_size = loaded_action_size
                    self.max_links = loaded_max_links
                    self.link_dim = loaded_link_dim
                    
                    # Rebuild models
                    self.model = AdvancedPhaseDQN(self.max_links, self.link_dim, self.num_phases).to(device)
                    self.target_model = AdvancedPhaseDQN(self.max_links, self.link_dim, self.num_phases).to(device)
            
            # Now load model state dicts
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.target_model.load_state_dict(torch.load(target_model_path, map_location=device))
            self.model.to(device)
            self.target_model.to(device)

            # Load optimizer - safer to re-initialize with loaded learning rate then load state
            loaded_alpha = hyperparams.get('alpha', self.alpha)
            
            # Re-initialize optimizer with loaded learning rate
            self.optimizer = optim.Adam(self.model.parameters(), lr=loaded_alpha, weight_decay=1e-5)
            
            # Now load the saved optimizer state dict
            if os.path.exists(optimizer_path):
                self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
                adv_logger.info(f"Optimizer state loaded (re-initialized with alpha={loaded_alpha}).")
            else:
                adv_logger.warning("Optimizer state file not found. Initialized fresh optimizer.")

            # Update agent parameters from loaded hyperparams
            self.epsilon = hyperparams.get('epsilon', self.epsilon)
            self.gamma = hyperparams.get('gamma', self.gamma)
            self.alpha = loaded_alpha  # Already extracted above
            self.train_step = hyperparams.get('train_step', self.train_step)
            self.epsilon_decay = hyperparams.get('epsilon_decay', self.epsilon_decay)
            self.epsilon_min = hyperparams.get('epsilon_min', self.epsilon_min)
            self.target_update_freq = hyperparams.get('target_update_freq', self.target_update_freq)
            self.batch_size = hyperparams.get('batch_size', self.batch_size)
            self.priority_alpha = hyperparams.get('priority_alpha', self.priority_alpha)
            self.priority_beta = hyperparams.get('priority_beta', self.priority_beta)
            
            # Load normalization statistics if they exist
            if os.path.exists(normalization_path):
                try:
                    norm_stats = torch.load(normalization_path, map_location=device)
                    self.stats_count = norm_stats['stats_count']
                    self.feature_means = norm_stats['feature_means']
                    self.feature_variances = norm_stats['feature_variances']
                    adv_logger.info(f"Loaded normalization statistics: count={self.stats_count}")
                except Exception as e:
                    adv_logger.warning(f"Error loading normalization statistics: {e}")
                    # Initialize fresh statistics
                    self.stats_count = torch.tensor(0, device=device)
                    self.feature_means = torch.zeros(4, device=device)
                    self.feature_variances = torch.zeros(4, device=device)
            else:
                adv_logger.info("No normalization statistics found, initializing fresh ones")
                # Initialize fresh statistics
                self.stats_count = torch.tensor(0, device=device)
                self.feature_means = torch.zeros(4, device=device)
                self.feature_variances = torch.zeros(4, device=device)

            # Set model modes correctly after loading
            self.model.train()
            self.target_model.eval()

            adv_logger.info(f"AdvancedAgent (Phase-based) state loaded successfully for {self.tls_id}.")

        except FileNotFoundError:
            adv_logger.error(f"Error loading AdvancedAgent state: File not found during load attempt in {directory_path}")
        except Exception as e:
            adv_logger.error(f"Error loading AdvancedAgent state for {self.tls_id}: {e}", exc_info=True)
