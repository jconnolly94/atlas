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


class AdvancedTrafficNetworkDQN(nn.Module):
    """Advanced Deep Q-Network for traffic signal control with sophisticated architecture."""

    def __init__(self, max_links=16, link_dim=6, output_dim=2):
        """Initialize network architecture.

        Args:
            max_links: Maximum number of links to support
            link_dim: Dimensionality of each link's feature vector
            output_dim: Output dimension for each link (green or red)
        """
        super(AdvancedTrafficNetworkDQN, self).__init__()

        # Embedding dimensions
        self.link_embedding_dim = 16
        self.signal_embedding_dim = 16
        self.max_links = max_links

        # Encoders
        self.link_encoder = AdvancedLinkEncoder(link_dim, self.link_embedding_dim)
        self.signal_encoder = AdvancedSignalStateEncoder(self.signal_embedding_dim)

        # Link selection network with additional layers
        self.link_selector = nn.Sequential(
            nn.Linear(self.link_embedding_dim + self.signal_embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)  # Scores for each link
        )

        # Action selection network with additional layers
        self.action_selector = nn.Sequential(
            nn.Linear(self.link_embedding_dim + self.signal_embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, output_dim)  # Q-values for each possible state (G, r)
        )

        # Historical trend analyzer for long-term patterns
        self.trend_analyzer = nn.Sequential(
            nn.Linear(self.link_embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)  # Trend features
        )

    def forward(self, link_features, signal_state):
        """Forward pass through the network with improved architecture.

        Args:
            link_features: Tensor of link features [batch_size, num_links, link_dim]
            signal_state: List of signal state strings (e.g., ["GrGr", "rGrG"])

        Returns:
            link_scores: Scores for selecting each link [batch_size, num_links]
            action_values: Q-values for actions on each link [batch_size, num_links, output_dim]
            trend_features: Features representing traffic trends [batch_size, num_links, 8]
        """
        batch_size = link_features.shape[0]
        num_links = link_features.shape[1]
        device = link_features.device # Get device from input tensor

        # Ensure we have at least one sample to process
        if batch_size == 0 or num_links == 0:
            # Return empty tensors of appropriate shapes if no data
            return (torch.zeros(0, num_links, device=device),
                    torch.zeros(0, num_links, 2, device=device),  # 2 for G and r
                    torch.zeros(0, num_links, 8, device=device))  # 8 trend features

        # Encode link features
        link_embeddings = self.link_encoder(link_features)  # [batch_size, num_links, link_embedding_dim]

        try:
            # Encode signal states with attention
            signal_embeddings = self.signal_encoder(signal_state)  # [batch_size, max_signal_len, signal_embedding_dim]

            # Check signal embeddings shape and device for safety
            if signal_embeddings.dim() < 3 or signal_embeddings.size(0) != batch_size:
                 adv_logger.warning(f"Unexpected signal embedding shape: {signal_embeddings.shape}, expected batch size {batch_size}. Using zeros.")
                 signal_embeddings = torch.zeros(batch_size, num_links, self.signal_embedding_dim, device=device)
            else:
                 # Ensure signal embeddings are on the same device
                 signal_embeddings = signal_embeddings.to(device)

            # Adjust num_links based on actual signal embedding length if shorter
            signal_len = signal_embeddings.size(1)
            effective_num_links = min(num_links, signal_len)

        except Exception as e:
            # Handle any unexpected errors during signal encoding
            adv_logger.error(f"Error encoding signal states: {e}", exc_info=True)
            # Create a default tensor of zeros
            signal_embeddings = torch.zeros(batch_size, num_links, self.signal_embedding_dim, device=device)
            effective_num_links = num_links


        # Concatenate link embeddings with corresponding signal state embeddings
        # Ensure concatenation happens correctly even if signal_embeddings are shorter
        combined_features_list = []
        for b in range(batch_size):
            link_batch_features = []
            for l in range(num_links):
                link_emb = link_embeddings[b, l]
                if l < effective_num_links:
                    sig_emb = signal_embeddings[b, l]
                else:
                    # Pad with zeros if link index exceeds signal length
                    sig_emb = torch.zeros(self.signal_embedding_dim, device=device)
                combined = torch.cat([link_emb, sig_emb])
                link_batch_features.append(combined)
            combined_features_list.append(torch.stack(link_batch_features)) # Stack features for links in this batch item

        # Stack features across the batch
        combined_features = torch.stack(combined_features_list) # [batch_size, num_links, combined_dim]


        # Get link selection scores
        link_scores = self.link_selector(combined_features).squeeze(-1)  # [batch_size, num_links]

        # Get action values for each link
        action_values = self.action_selector(combined_features)  # [batch_size, num_links, output_dim]

        # Get trend features for historical analysis
        trend_features = self.trend_analyzer(link_embeddings)  # [batch_size, num_links, 8]

        return link_scores, action_values, trend_features


class AdvancedAgent(Agent):
    """Advanced agent using Deep Q-Learning with sophisticated architecture for lane-level traffic control."""

    # Default configuration
    DEFAULT_CONFIG = {
        "alpha": 0.0005,          # Lower learning rate for stability
        "gamma": 0.98,            # Higher discount factor for long-term planning
        "epsilon": 0.95,           # INCREASE: Start with more randomness
        "epsilon_decay": 0.9997,   # DECREASE DECAY RATE: Slow down decay significantly
        "epsilon_min": 0.05,       # Keep minimum low for long-term
        "batch_size": 64,         # Larger batch size for better gradient estimates
        "memory_size": 50000,     # Much larger memory for diverse experiences
        "target_update_freq": 250,  # Less frequent updates for stability
        "double_dqn": True,       # Use double DQN for more stable learning
        "prioritized_replay": True,  # Use prioritized experience replay
        "priority_alpha": 0.6,    # Priority exponent for PER
        "priority_beta": 0.4      # Importance sampling exponent for PER
    }

    def __init__(self, tls_id, network, conflict_detector, alpha=0.0005, gamma=0.98, epsilon=0.9,
                 epsilon_decay=0.9997, epsilon_min=0.2, batch_size=64,
                 memory_size=50000, target_update_freq=250, double_dqn=True,
                 prioritized_replay=True, priority_alpha=0.6, priority_beta=0.4):
        """Initialize AdvancedAgent with lane-level control capabilities.
        # ... (rest of args description) ...
            priority_alpha: Exponent alpha for prioritized experience replay.
            priority_beta: Exponent beta for prioritized experience replay importance sampling.
        """
        super().__init__(tls_id, network)
        self.network = network
        self.conflict_detector = conflict_detector

        # Maximum number of links and features per link
        self.max_links = 26  # Support more links than standard DQN
        self.link_dim = 6    # Add an extra feature for historical trend

        # Possible link states (G for green, r for red)
        self.link_states = ['G', 'r']  # Simplified to just green and red

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
        self.priority_beta_increment = 0.001  # Consider making this configurable too
        self.priority_epsilon = 0.01         # Small constant to avoid zero priority

        # Value to use for masking invalid actions (instead of 1e9)
        self.masking_value = 100.0  # More moderate value to avoid numerical instability
        
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

        # Create main and target networks with advanced architecture
        self.model = AdvancedTrafficNetworkDQN(self.max_links, self.link_dim, len(self.link_states)).to(self.device)
        self.target_model = AdvancedTrafficNetworkDQN(self.max_links, self.link_dim, len(self.link_states)).to(self.device)

        # Copy weights from main to target network
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval() # Target network should be in eval mode

        # Optimizer with weight decay for regularization
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha, weight_decay=1e-5)

        # Target network update frequency
        self.train_step = 0

        # Track last action and state for reward calculation
        self.last_action = None
        self.last_state = None

        # Traffic pattern history for advanced analysis
        self.traffic_history = defaultdict(lambda: deque(maxlen=50))  # Store last 50 observations per link

        # Performance tracking for adaptive exploration
        self.performance_history = deque(maxlen=20)  # Store last 20 rewards
        
        # Initialize running statistics for online normalization
        self.stats_count = torch.tensor(0, device=self.device)
        # Create tensors for each feature type: [queue_length, waiting_time, vehicle_count, time_since_last_change]
        self.feature_means = torch.zeros(4, device=self.device)
        self.feature_variances = torch.zeros(4, device=self.device)

    def _create_action_mask(self, link_states, signal_state_str, num_links=None):
        """Creates an action mask for the given state.
        
        Args:
            link_states: List of dictionaries containing link state information
            signal_state_str: String representing the current signal state
            num_links: Optional number of links to use (defaults to length of link_states)
            
        Returns:
            Tensor with shape [num_links, len(self.link_states)] where 1 indicates valid actions
                and 0 indicates invalid actions
        """
        if num_links is None:
            num_links = len(link_states)
            
        # Create mask tensor (default: all actions are valid)
        mask = torch.ones(num_links, len(self.link_states), device=self.device)
        
        # If no links or no signal state, return an empty mask
        if num_links == 0 or not signal_state_str:
            adv_logger.warning(f"Cannot create mask: No links or signal state for {self.tls_id}")
            return mask
            
        # Validate and mark invalid actions
        for internal_idx in range(min(num_links, len(link_states))):
            # Add safety check for link structure
            current_link_data = link_states[internal_idx]
            if not isinstance(current_link_data, dict) or 'index' not in current_link_data:
                adv_logger.warning(f"Invalid link state structure during masking for {self.tls_id} at index {internal_idx}. Assuming invalid.")
                mask[internal_idx, :] = 0  # Mark all actions for this link as invalid
                continue  # Skip to next link
                
            link_idx_sumo = current_link_data['index']  # Get SUMO index
            for action_idx, new_state in enumerate(self.link_states):  # self.link_states = ['G', 'r']
                action_tuple = (link_idx_sumo, new_state)
                if not self._is_action_valid(action_tuple, signal_state_str):
                    mask[internal_idx, action_idx] = 0  # Mark as invalid
                    
        return mask
    
    def _apply_mask_to_q_values(self, q_values, mask):
        """Applies an action mask to Q-values.
        
        Args:
            q_values: Tensor of Q-values [batch_size, num_links, num_actions]
            mask: Tensor with shape [batch_size, num_links, num_actions] where 1 indicates valid actions
            
        Returns:
            Masked Q-values with invalid actions set to a large negative value
        """
        # Ensure mask is properly broadcast to match q_values shape
        if mask.dim() < q_values.dim():
            if mask.dim() == 2 and q_values.dim() == 3:
                # Add batch dimension if needed
                mask = mask.unsqueeze(0).expand_as(q_values)
                
        # Clone Q-values to avoid modifying the original
        masked_q_values = q_values.clone()
        
        # Apply mask - use a more moderate negative value to avoid numerical instability
        masked_q_values = masked_q_values - (self.masking_value * (1 - mask))
        
        return masked_q_values
        
    def _is_action_valid(self, action_tuple: Tuple[int, str], current_signal_state: str) -> bool:
        """Checks if an action conflicts with the current signal state.

        Args:
            action_tuple: Tuple of (link_index, new_state) representing the action to validate
            current_signal_state: String representing the current signal state
            
        Returns:
            True if the action is valid (doesn't create conflicts), False otherwise
        """
        link_idx_sumo, new_state = action_tuple
        # Setting to 'r' is always valid from a conflict perspective
        if new_state == 'r':
            return True
        # Check validity only when setting to 'G'
        if new_state == 'G':
            # Ensure conflict detector is available
            if not hasattr(self, 'conflict_detector') or self.conflict_detector is None:
                 adv_logger.error(f"Conflict detector not available in _is_action_valid for {self.tls_id}. Assuming action is invalid.")
                 return False # Assume invalid if detector is missing
            # Ensure current_signal_state is valid and link_idx_sumo is within bounds
            if not isinstance(current_signal_state, str) or link_idx_sumo < 0 or link_idx_sumo >= len(current_signal_state):
                 adv_logger.warning(f"Invalid signal state or link index ({link_idx_sumo}) in _is_action_valid for {self.tls_id}.")
                 return False # Cannot validate
            # Find SUMO indices of currently green links
            currently_green_indices = {
                i for i, char in enumerate(current_signal_state) if char in 'Gg'
            }
            # Get links that conflict with the target link
            conflicting_links = self.conflict_detector.get_conflicting_links(self.tls_id, link_idx_sumo)
            # Check if any currently green link conflicts with the target link
            if any(g_idx in conflicting_links for g_idx in currently_green_indices):
                # adv_logger.debug(f"Action {(link_idx_sumo, new_state)} invalid: Conflicts with green links {currently_green_indices.intersection(conflicting_links)}")
                return False # Conflict detected!
        # If no conflicts found for 'G', or action is not 'G', it's valid
        return True

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
        """
        # Extract link states and current signal state
        link_states = state.get('link_states', []) # Use .get for safety
        current_signal_state = state.get('current_signal_state', "")

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
            
            # Assemble the feature vector
            features = [
                norm_queue,
                norm_waiting,
                norm_vehicles,
                norm_time_change,
                is_green,
                historical_trend
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
            link_features.append([0.0] * self.link_dim)

        # Truncate if too many links (shouldn't happen if max_links is large enough)
        link_features = link_features[:self.max_links]

        return np.array(link_features), current_signal_state

    def choose_action(self, state):
        """Choose an action based on the current state with advanced exploration strategies.
        # ... (rest of implementation) ...
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
        exploration_rate = self.epsilon # Default
        if len(self.performance_history) >= 5: # Require a minimum history
            # Calculate the average of recent rewards
            recent_rewards = list(self.performance_history)
            recent_rewards_subset = recent_rewards[-min(5, len(recent_rewards)):]
            if recent_rewards_subset: # Avoid division by zero
                 avg_reward = sum(recent_rewards_subset) / len(recent_rewards_subset)
                 # If rewards are consistently poor, increase exploration
                 if avg_reward < -0.5: # Threshold might need tuning
                     exploration_rate = min(1.0, self.epsilon * 1.2)

        # Exploration: choose random action with probability epsilon
        if np.random.rand() <= exploration_rate:
            # Exploration: choose VALID random action
            valid_action_found = False
            attempts = 0
            max_attempts = 50 # Prevent infinite loops if something is wrong
            # Need original link_states list from the input state dictionary
            original_link_states = state.get('link_states', [])
            num_actual_links = len(original_link_states)
            current_signal_state_str = state.get('current_signal_state', "")
            action = None # Initialize action to None (fallback)
            if num_actual_links == 0 or not current_signal_state_str:
                 adv_logger.warning(f"Cannot explore: No links or signal state for {self.tls_id}")
                 # action remains None
            else:
                while not valid_action_found and attempts < max_attempts:
                    attempts += 1
                    # Choose a random link index *from the actual links available*
                    random_internal_idx = random.randrange(num_actual_links)
                    # Get the corresponding SUMO link index, checking link structure
                    current_link_data = original_link_states[random_internal_idx]
                    if not isinstance(current_link_data, dict) or 'index' not in current_link_data:
                         adv_logger.warning(f"Invalid link state structure during exploration for {self.tls_id}. Skipping attempt.")
                         continue # Try next attempt
                    link_idx_sumo = current_link_data['index']
                    # Choose a random target state ('G' or 'r')
                    new_state = random.choice(self.link_states)
                    # Construct the action tuple using SUMO index
                    candidate_action = (link_idx_sumo, new_state)
                    # Check validity
                    if self._is_action_valid(candidate_action, current_signal_state_str):
                        action = candidate_action
                        valid_action_found = True
                    # else:
                        # adv_logger.debug(f"Exploration rejected invalid action: {candidate_action}")
                if not valid_action_found:
                     adv_logger.warning(f"Exploration failed to find valid action after {max_attempts} attempts for {self.tls_id}. Choosing no-op (None).")
                     action = None # Explicitly set to None if loop fails
        else:
            # Exploitation: Use the advanced DQN to select the best action
            link_tensor = torch.FloatTensor(link_features_np).unsqueeze(0).to(self.device)

            self.model.eval() # Set model to evaluation mode for inference
            with torch.no_grad():
                # Get link scores and action values from advanced model
                link_scores, action_values, trend_features = self.model(link_tensor, [signal_state_str])

            self.model.train() # Set model back to training mode

            # --- Action Masking ---
            original_link_states = state.get('link_states', [])
            num_actual_links = len(original_link_states)
            # Use the signal state string obtained from preprocessing earlier
            current_signal_state_str = signal_state_str

            action = None # Initialize action to None (fallback)

            if num_actual_links == 0 or not current_signal_state_str:
                 adv_logger.warning(f"Cannot exploit: No links or signal state for {self.tls_id}")
                 # action remains None
            else:
                # Create action mask using the helper method
                action_mask = self._create_action_mask(original_link_states, current_signal_state_str, num_actual_links)

                # Apply mask to Q-values (only for actual links)
                # Ensure slicing is correct for potentially padded action_values
                masked_q_values = action_values[0, :num_actual_links, :].clone() # Work on a copy [num_actual_links, num_actions]

                # Subtract a large number from invalid actions to effectively remove them from argmax
                masked_q_values = masked_q_values - (self.masking_value * (1 - action_mask)) # Mask is already on device

                # Find the best valid action based on masked Q-values
                # Find the best action across *all* valid links and *all* action types
                try:
                    # Check if all actions are masked
                    if torch.all(masked_q_values < -0.9 * self.masking_value):
                        adv_logger.warning(f"All actions masked during exploitation for {self.tls_id}. Choosing no-op (None).")
                        action = None
                    else:
                        best_flat_idx = torch.argmax(masked_q_values).item()
                        # Convert flat index back to (link_internal_idx, action_idx)
                        best_link_idx_internal, best_action_idx = np.unravel_index(best_flat_idx, masked_q_values.shape)

                        # Map back to SUMO link index and state string
                        # Add safety check for link structure again before accessing 'index'
                        chosen_link_data = original_link_states[best_link_idx_internal]
                        if not isinstance(chosen_link_data, dict) or 'index' not in chosen_link_data:
                             adv_logger.error(f"Invalid link structure for chosen best link index {best_link_idx_internal}. Cannot determine action.")
                             action = None
                        else:
                            link_index_to_change = chosen_link_data['index']
                            new_state = self.link_states[best_action_idx]
                            action = (link_index_to_change, new_state)

                except ValueError as e:
                     # Catch potential errors if masked_q_values is empty
                     adv_logger.error(f"Error finding best action during exploitation for {self.tls_id} (ValueError): {e}. Falling back to None.")
                     action = None # Fallback if argmax fails

            # --- End Action Masking ---

        # Update epsilon with adaptive decay logic (corrected logic)
        if len(self.performance_history) >= 10:
             recent_rewards = list(self.performance_history)
             # Ensure indices are valid
             split_point = max(0, len(recent_rewards) - 5)
             recent_avg = sum(recent_rewards[split_point:]) / max(1, len(recent_rewards) - split_point)
             older_avg = sum(recent_rewards[:split_point]) / max(1, split_point)

             # Check if performance history is sufficient for comparison
             if len(recent_rewards) - split_point > 0 and split_point > 0:
                 if recent_avg > older_avg:
                     effective_decay = self.epsilon_decay * 1.001 # Slower decay
                 else:
                     effective_decay = self.epsilon_decay * 0.999 # Faster decay
                 self.epsilon = max(self.epsilon_min, self.epsilon * effective_decay)
             else: # Not enough distinct history for comparison, use standard decay
                 self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        else:
             # Standard decay if not enough history
             self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


        # Store last action for learning
        self.last_action = action
        return action

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory with prioritization. (Corrected Mapping)"""
        if action is None or state is None or next_state is None:
            adv_logger.warning("Skipping remember due to None input.")
            return

        try:
            # Preprocess states (creates padded arrays of size max_links)
            link_features, signal_state = self._preprocess_state(state)
            next_link_features, next_signal_state = self._preprocess_state(next_state)

            # Extract link index (SUMO index) and new state from the action tuple
            link_idx_sumo, new_state = action

            # --- START: Robust Positional Index Finding ---
            # Initialize internal index to invalid value
            link_internal_idx = -1
            
            # Get the original list of link state dictionaries from the state
            original_link_states = state.get('link_states', [])
            
            # Iterate through the original_link_states list, but only consider entries
            # up to the self.max_links limit (because features beyond this weren't stored)
            for i in range(min(len(original_link_states), self.max_links)):
                # Check if this link's index matches the SUMO index from the action
                if original_link_states[i].get('index') == link_idx_sumo:
                    link_internal_idx = i
                    break
            
            # Error Handling: Check if a valid mapping was found
            if link_internal_idx == -1:
                # The link_idx_sumo wasn't found in the processed links
                adv_logger.error(f"Failed to map SUMO link index {link_idx_sumo} to internal position "
                                 f"for TLS {self.tls_id}. Link not found or beyond max_links limit. "
                                 f"Skipping experience storage.")
                return  # Skip storing this experience as it cannot be learned from correctly
            # --- END: Robust Positional Index Finding ---

            # Find the index of the new state in our defined link_states list ['G', 'r']
            action_idx = self.link_states.index(new_state) if new_state in self.link_states else 0

            # Create experience tuple (using the validated positional index)
            # Store the original link states in the experience for action masking during replay
            original_next_link_states = next_state.get('link_states', [])

            # Create experience tuple (using the validated positional index)
            experience = (
                link_features,              # Numpy array [max_links, link_dim]
                signal_state,               # String
                (link_internal_idx, action_idx), # Tuple (VALIDATED POSITIONAL idx, action type idx)
                float(reward),              # Ensure reward is float
                next_link_features,         # Numpy array [max_links, link_dim]
                next_signal_state,          # String
                bool(done),                 # Ensure done is bool
                original_link_states,       # Original link states for action masking
                original_next_link_states   # Original next link states for action masking
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


    def learn(self, state, action, next_state, done):
        """Learn from experience using advanced deep Q-learning.
        # ... (rest of implementation) ...
        """
        if action is None:
            return

        # Calculate reward using agent's own method
        reward, _ = self.calculate_reward(state, action, next_state)

        # Store reward in performance history
        self.performance_history.append(reward)

        # Store experience in memory (handles preprocessing)
        self.remember(state, action, reward, next_state, done)

        # Determine required memory size based on replay type
        min_memory_for_replay = self.batch_size
        if self.prioritized_replay:
            # PER might technically work with less, but batch_size is practical minimum
            current_memory_size = len(self.memory)
        else:
            current_memory_size = len(self.memory)

        # Only learn if we have enough samples
        if current_memory_size < min_memory_for_replay:
            return

        # Perform experience replay
        self._replay()


    def _replay(self):
        """Perform experience replay with advanced techniques."""
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
            batch_priorities = priorities[indices]

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

        # --- Process Batch ---
        # Unpack the batch, converting numpy arrays to tensors on the correct device
        link_features_batch = torch.FloatTensor(np.array([t[0] for t in batch])).to(self.device)
        signal_state_batch = [t[1] for t in batch] # List of strings
        actions_batch = [t[2] for t in batch] # List of tuples (link_internal_idx, action_idx)
        rewards_batch = torch.FloatTensor([t[3] for t in batch]).to(self.device)
        next_link_features_batch = torch.FloatTensor(np.array([t[4] for t in batch])).to(self.device)
        next_signal_state_batch = [t[5] for t in batch] # List of strings
        dones_batch = torch.FloatTensor([t[6] for t in batch]).to(self.device)
        original_link_states_batch = [t[7] for t in batch] # List of original link states
        original_next_link_states_batch = [t[8] for t in batch] # List of original next link states

        # Separate link indices and action indices into tensors
        link_indices = torch.LongTensor([a[0] for a in actions_batch]).to(self.device)
        action_indices = torch.LongTensor([a[1] for a in actions_batch]).to(self.device)

        # --- Get Current Q-values ---
        # Shape: [batch_size, num_links, num_actions]
        _, current_q_all_actions, _ = self.model(link_features_batch, signal_state_batch)

        # Select the Q-value for the action actually taken in the experience
        # Gather needs indices of shape [batch_size, 1] for actions
        # First, gather Q-values for the specific link index acted upon
        # current_q_all_actions shape: [batch_size, max_links, num_actions]
        # link_indices shape: [batch_size] -> needs unsqueeze to [batch_size, 1] for gather
        # And also expand it to match the num_actions dimension for gather
        link_indices_expanded = link_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, current_q_all_actions.size(2))
        # Select Q-values for the specific link: result shape [batch_size, 1, num_actions]
        current_q_selected_link = torch.gather(current_q_all_actions, 1, link_indices_expanded.to(torch.int64))
        current_q_selected_link = current_q_selected_link.squeeze(1) # Shape: [batch_size, num_actions]

        # Now gather the Q-value for the specific action taken on that link
        # action_indices shape: [batch_size] -> needs unsqueeze to [batch_size, 1]
        current_q = torch.gather(current_q_selected_link, 1, action_indices.unsqueeze(1)).squeeze(1) # Shape: [batch_size]


        # --- Target Q-values with Double DQN and Action Masking ---
        with torch.no_grad():
            # 1. Get action values from the online model for next state
            _, next_action_values_online, _ = self.model(
                next_link_features_batch, next_signal_state_batch
            )
            
            # Apply action masking to next-state Q-values to ensure consistency
            batch_size = next_action_values_online.shape[0]
            
            # Create a list to store masked action values for each batch item
            masked_next_action_values_online_list = []
            
            # Apply action masking to each item in the batch
            for batch_idx in range(batch_size):
                # Get original link states and signal state for this batch item
                next_links = original_next_link_states_batch[batch_idx]
                next_signal = next_signal_state_batch[batch_idx]
                
                # Create mask for this batch item
                num_links = min(len(next_links), next_action_values_online.shape[1])
                action_mask = self._create_action_mask(next_links, next_signal, num_links)
                
                # Apply mask to this batch item's Q-values
                masked_item_q_values = self._apply_mask_to_q_values(
                    next_action_values_online[batch_idx:batch_idx+1, :num_links, :],
                    action_mask
                )
                
                masked_next_action_values_online_list.append(masked_item_q_values.squeeze(0))
            
            # For each batch item, find the best valid action according to online network
            best_next_link_indices = []
            best_next_action_indices = []
            
            for batch_idx, masked_q_values in enumerate(masked_next_action_values_online_list):
                # Skip if all actions are invalid for this item (extremely rare but possible)
                if torch.all(masked_q_values < -0.9 * self.masking_value):
                    # If all actions invalid, use a default action (first link, 'r' action)
                    best_link_idx = 0
                    best_action_idx = self.link_states.index('r')  # Usually 1 for 'r'
                else:
                    # Find the best valid action
                    best_flat_idx = torch.argmax(masked_q_values).item()
                    best_link_idx, best_action_idx = np.unravel_index(
                        best_flat_idx, masked_q_values.shape
                    )
                
                best_next_link_indices.append(best_link_idx)
                best_next_action_indices.append(best_action_idx)
            
            # Convert to tensors
            best_next_link_indices = torch.tensor(best_next_link_indices, device=self.device)
            best_next_action_indices = torch.tensor(best_next_action_indices, device=self.device)
            
            # 2. Evaluate these selected actions using the target model
            _, next_action_values_target, _ = self.target_model(
                next_link_features_batch, next_signal_state_batch
            )
            
            # Construct indices for gathering from target network
            # Create a batch index for gather
            batch_indices = torch.arange(batch_size, device=self.device)
            
            # Gather Q-values from target network for the link and action selected by online network
            q_target_next = next_action_values_target[
                batch_indices, best_next_link_indices, best_next_action_indices
            ]  # Shape: [batch_size]

            # Calculate final target Q value: R + gamma * Q_target_next * (1 - done)
            target_q = rewards_batch + (1 - dones_batch) * self.gamma * q_target_next


        # Compute loss (TD Error) element-wise
        td_errors = target_q - current_q # Keep shape [batch_size]
        loss = (importance_weights * (td_errors ** 2)).mean() # Apply IS weights before averaging MSE

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

    def calculate_reward(self, state: Dict[str, Any], action: Optional[Tuple[int, str]],
                         next_state: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate reward based on minimizing traffic pressure.
        Pressure is approximated by the sum of queue lengths on incoming lanes.
        Args:
            state: Previous state dictionary (unused in this version).
            action: Action taken tuple (unused in this version).
            next_state: Resulting state dictionary.
        Returns:
            Tuple[float, Dict[str, float]]: Calculated reward (-pressure) and components.
        """
        # --- Input Validation ---
        if not isinstance(next_state, dict):
            # Use logger associated with the instance if available, otherwise default
            logger_instance = getattr(self, 'adv_logger', logging.getLogger(__name__))
            logger_instance.warning(f"Invalid next_state for reward calc (TLS: {self.tls_id})")
            return 0.0, {}
        # Get link states from next_state
        link_states = next_state.get('link_states', [])
        if not link_states:
            # No links, pressure is effectively zero
            return 0.0, {'total_queue_length': 0.0, 'pressure': 0.0, 'reward': 0.0}
        # Calculate pressure = sum of queue lengths on all controlled links
        total_queue_length = sum(link.get('queue_length', 0.0) for link in link_states)
        pressure = total_queue_length # Using queue length as pressure proxy
        # Scale the reward (negative pressure)
        # Aim for rewards roughly in [-10, 0]. Max queue sum might be ~100-200?
        PRESSURE_SCALE_FACTOR = 20.0 # Adjust based on observed queue sums
        # Reward is negative pressure (we want to minimize pressure)
        reward = -pressure / PRESSURE_SCALE_FACTOR
        # Ensure reward is not excessively large/small
        reward = max(-10.0, min(reward, 0.0)) # Clip reward
        # Return reward and components for analysis
        components = {
            'total_queue_length': total_queue_length,
            'pressure': pressure,
            'reward': reward
        }
        return reward, components

    # --- save_state and load_state methods ---
    def save_state(self, directory_path: str):
        """Saves the AdvancedAgent's state to the specified directory."""
        super().save_state(directory_path) # Creates directory via base class call
        #adv_logger.info(f"Saving AdvancedAgent state for {self.tls_id} to {directory_path}")

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
                # Add other config if needed
            }
            with open(hyperparams_path, 'w') as f:
                json.dump(hyperparams, f, indent=4)


            #adv_logger.info(f"AdvancedAgent state saved successfully for {self.tls_id}.")

        except Exception as e:
            adv_logger.error(f"Error saving AdvancedAgent state for {self.tls_id}: {e}", exc_info=True)


    def load_state(self, directory_path: str):
        """Loads the AdvancedAgent's state from the specified directory."""
        adv_logger.info(f"Loading AdvancedAgent state for {self.tls_id} from {directory_path}")

        model_path = os.path.join(directory_path, 'model.pth')
        target_model_path = os.path.join(directory_path, 'target_model.pth')
        optimizer_path = os.path.join(directory_path, 'optimizer.pth')
        hyperparams_path = os.path.join(directory_path, 'hyperparams.json')
        normalization_path = os.path.join(directory_path, 'normalization.pth')

        # Check if essential files exist
        required_files = [model_path, target_model_path, optimizer_path, hyperparams_path]
        if not all(os.path.exists(p) for p in required_files):
            adv_logger.warning(f"Cannot load AdvancedAgent state for {self.tls_id}: Required file(s) not found in {directory_path}. Starting fresh.")
            # Re-initialize relevant parts or rely on __init__ defaults
            self.__init__(self.tls_id, self.network, **self.DEFAULT_CONFIG) # Re-init with defaults might be safest
            return

        try:
            # Determine device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            adv_logger.info(f"Loading state onto device: {device}")

            # Load models - ensure model instances exist first
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.target_model.load_state_dict(torch.load(target_model_path, map_location=device))
            self.model.to(device)
            self.target_model.to(device)

            # Load optimizer state - safer to re-initialize with loaded LR then load state
            # Load learning rate first
            loaded_alpha = self.alpha # Default to current alpha
            try:
                with open(hyperparams_path, 'r') as f:
                    loaded_params = json.load(f)
                    loaded_alpha = loaded_params.get('alpha', self.alpha)
            except Exception as e:
                adv_logger.warning(f"Could not read alpha from hyperparams file, using current: {e}")

            # Re-initialize optimizer with potentially loaded learning rate
            self.optimizer = optim.Adam(self.model.parameters(), lr=loaded_alpha, weight_decay=1e-5)
            # Now load the saved state dict
            self.optimizer.load_state_dict(torch.load(optimizer_path))
            adv_logger.info(f"Optimizer state loaded (re-initialized with alpha={loaded_alpha}).") 


            # Load hyperparameters from JSON
            with open(hyperparams_path, 'r') as f:
                hyperparams = json.load(f)
                self.epsilon = hyperparams.get('epsilon', self.epsilon)
                self.gamma = hyperparams.get('gamma', self.gamma)
                self.alpha = hyperparams.get('alpha', self.alpha) # Update alpha attribute
                self.train_step = hyperparams.get('train_step', self.train_step)
                self.epsilon_decay = hyperparams.get('epsilon_decay', self.epsilon_decay)
                self.epsilon_min = hyperparams.get('epsilon_min', self.epsilon_min)
                self.target_update_freq = hyperparams.get('target_update_freq', self.target_update_freq)
                self.batch_size = hyperparams.get('batch_size', self.batch_size) # Load batch size if needed later
                # Load PER params
                self.priority_alpha = hyperparams.get('priority_alpha', self.priority_alpha)
                self.priority_beta = hyperparams.get('priority_beta', self.priority_beta) # Load saved beta
            
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

            adv_logger.info(f"AdvancedAgent state loaded successfully for {self.tls_id}.")

        except FileNotFoundError:
            adv_logger.error(f"Error loading AdvancedAgent state: File not found during load attempt in {directory_path}")
        except Exception as e:
            adv_logger.error(f"Error loading AdvancedAgent state for {self.tls_id}: {e}", exc_info=True)
