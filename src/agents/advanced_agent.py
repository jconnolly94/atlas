# --- START OF FILE advanced_agent.py ---

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

        # Pad sequences to the same length if necessary before stacking
        # Note: If all state strings have the same length, padding isn't needed.
        # If lengths vary, use torch.nn.utils.rnn.pad_sequence
        try:
             # Simple stacking assuming equal length for now
             return torch.stack(batch_embeddings)  # [batch_size, seq_len, embedding_dim]
        except RuntimeError as e:
             adv_logger.error(f"Error stacking signal embeddings (lengths might differ?): {e}")
             # Implement padding if needed based on error message
             # Example using pad_sequence (adjust batch_first as needed):
             # padded_embeddings = torch.nn.utils.rnn.pad_sequence(batch_embeddings, batch_first=True, padding_value=0.0)
             # return padded_embeddings
             # For now, return an empty tensor or raise error to highlight the issue
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
        "epsilon": 0.9,           # Very high exploration rate
        "epsilon_decay": 0.9997,  # Very slow decay to keep exploring longer
        "epsilon_min": 0.2,       # Higher minimum to maintain some exploration
        "batch_size": 64,         # Larger batch size for better gradient estimates
        "memory_size": 50000,     # Much larger memory for diverse experiences
        "target_update_freq": 250,  # Less frequent updates for stability
        "double_dqn": True,       # Use double DQN for more stable learning
        "prioritized_replay": True,  # Use prioritized experience replay
        "priority_alpha": 0.6,    # Priority exponent for PER
        "priority_beta": 0.4      # Importance sampling exponent for PER
    }

    def __init__(self, tls_id, network, alpha=0.0005, gamma=0.98, epsilon=0.9,
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

        # Maximum number of links and features per link
        self.max_links = 24  # Support more links than standard DQN
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
        # ... (rest of implementation) ...
        """
        # Extract link states and current signal state
        link_states = state.get('link_states', []) # Use .get for safety
        current_signal_state = state.get('current_signal_state', "")

        # Convert link states to a tensor with additional features
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

            # Use .get with defaults for safety
            features = [
                link.get('queue_length', 0) / 30.0,
                link.get('waiting_time', 0) / 1000.0,
                link.get('vehicle_count', 0) / 30.0,
                link.get('time_since_last_change', 0) / 200.0,
                is_green,
                historical_trend
            ]
            link_features.append(features)

            # Update traffic history for this link
            self.traffic_history[link_index].append({
                'queue_length': link.get('queue_length', 0),
                'waiting_time': link.get('waiting_time', 0),
                'vehicle_count': link.get('vehicle_count', 0)
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
            # Advanced exploration strategy: weighted sampling based on metrics
            weights = []
            valid_link_indices = [] # Store original indices corresponding to weights
            for i, link in enumerate(link_states): # Iterate original links before padding
                queue_weight = link.get('queue_length', 0) + 0.1  # Avoid zero weight
                waiting_weight = link.get('waiting_time', 0) / 100.0 + 0.1  # Scale waiting time

                # Get historical trend from traffic history
                link_index = link.get('index', -1)
                if link_index == -1: continue # Skip invalid links

                trend_weight = 0.1  # Default weight
                if link_index in self.traffic_history and len(self.traffic_history[link_index]) > 1:
                    queue_history = [record.get('queue_length', 0) for record in self.traffic_history[link_index]]
                    if len(queue_history) >= 2:
                        recent_trend = queue_history[-1] - queue_history[0]
                        trend_weight = max(0.1, recent_trend / 5.0 + 0.1)

                combined_weight = (queue_weight * 0.5) + (waiting_weight * 0.4) + (trend_weight * 0.1)
                weights.append(combined_weight)
                valid_link_indices.append(i) # Track the original index in link_states

            if not weights: # No valid links found
                adv_logger.warning(f"No valid links found during exploration for {self.tls_id}.")
                return None

            # Normalize weights to probabilities
            total_weight = sum(weights)
            if total_weight > 1e-6: # Use small epsilon for float comparison
                probs = [w / total_weight for w in weights]
                chosen_idx_in_valid = np.random.choice(len(valid_link_indices), p=probs)
                original_link_states_idx = valid_link_indices[chosen_idx_in_valid]
            else: # If all weights are near zero, choose uniformly from valid links
                original_link_states_idx = np.random.choice(valid_link_indices)

            # Get the actual link index and current state
            chosen_link = link_states[original_link_states_idx]
            link_index_to_change = chosen_link['index']
            current_signal_state_str = state['current_signal_state']
            if link_index_to_change >= len(current_signal_state_str):
                 adv_logger.warning(f"Chosen link index {link_index_to_change} out of bounds for signal state '{current_signal_state_str}'.")
                 return None # Cannot proceed

            current_link_char = current_signal_state_str[link_index_to_change]

            # Choose a new state different from the current one
            if current_link_char in 'Gg':
                new_state = 'r'  # Change green to red
            else:
                new_state = 'G'  # Change red/yellow to green

            action = (link_index_to_change, new_state)
        else:
            # Exploitation: Use the advanced DQN to select the best action
            link_tensor = torch.FloatTensor(link_features_np).unsqueeze(0).to(self.device)

            self.model.eval() # Set model to evaluation mode for inference
            with torch.no_grad():
                # Get link scores and action values from advanced model
                link_scores, action_values, trend_features = self.model(link_tensor, [signal_state_str])

            self.model.train() # Set model back to training mode

            # Use trend features to adjust link scores
            num_actual_links = len(link_states) # Number of real links before padding
            valid_links_to_consider = min(num_actual_links, self.max_links) # Consider scores only for actual links

            if valid_links_to_consider == 0:
                adv_logger.warning(f"No valid links to consider during exploitation for {self.tls_id}.")
                return None

            # Calculate priority scores that combine immediate value and trend
            priority_scores = link_scores[0, :valid_links_to_consider].clone().cpu().numpy() # Move to CPU for numpy ops

            # Adjust scores based on trend features if available
            if trend_features.numel() > 0 and trend_features.size(1) >= valid_links_to_consider:
                trend_cpu = trend_features[0, :valid_links_to_consider].cpu().numpy()
                for i in range(valid_links_to_consider):
                    trend_insight = trend_cpu[i].mean() # Simple mean of trend features
                    adjustment = trend_insight * 0.2 # Scale adjustment
                    priority_scores[i] += adjustment

            # Get the best link index within the valid range
            best_link_idx_in_processed = np.argmax(priority_scores) # Index relative to the 0..valid_links_to_consider range

            # Map back to the actual link's data
            chosen_link_data = link_states[best_link_idx_in_processed]
            link_index_to_change = chosen_link_data['index']

            # Get the best action for this chosen link (using the same index)
            best_action_idx = action_values[0, best_link_idx_in_processed].argmax().item()
            new_state = self.link_states[best_action_idx]

            # Advanced policy: Avoid changing if no need (check queue/wait)
            current_signal_state_str = state['current_signal_state']
            if link_index_to_change >= len(current_signal_state_str):
                 adv_logger.warning(f"Chosen link index {link_index_to_change} out of bounds during exploitation for signal state '{current_signal_state_str}'.")
                 return None # Cannot proceed

            current_link_char = current_signal_state_str[link_index_to_change]
            current_queue = chosen_link_data.get('queue_length', 0)
            current_waiting = chosen_link_data.get('waiting_time', 0)

            if ((new_state == 'G' and current_link_char in 'Gg') or \
                (new_state == 'r' and current_link_char in 'Rr')) and \
               current_queue < 3 and current_waiting < 100:

                # Find second best if the best is trivial
                if valid_links_to_consider > 1:
                    priority_scores[best_link_idx_in_processed] = -np.inf # Mask the best
                    second_best_idx_in_processed = np.argmax(priority_scores)

                    # Map back to the actual link's data
                    chosen_link_data = link_states[second_best_idx_in_processed]
                    link_index_to_change = chosen_link_data['index']

                    # Get the best action for this second-best link
                    best_action_idx = action_values[0, second_best_idx_in_processed].argmax().item()
                    new_state = self.link_states[best_action_idx]
                else:
                    # Only one link, no other choice
                    pass # Keep original action

            action = (link_index_to_change, new_state)

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
        """Store experience in replay memory with prioritization.
        # ... (rest of implementation) ...
        """
        if action is None or state is None or next_state is None:
            adv_logger.warning("Skipping remember due to None input.")
            return

        try:
            # Preprocess states
            link_features, signal_state = self._preprocess_state(state)
            next_link_features, next_signal_state = self._preprocess_state(next_state)

            # Extract link index and new state
            link_idx, new_state = action

            # Find the internal index (0 to max_links-1) of the link that was acted upon
            # This needs to map the *actual* link_index from SUMO to the *position* in our padded link_features array
            link_internal_idx = -1 # Default to invalid
            original_link_states = state.get('link_states', [])
            for i, link in enumerate(original_link_states):
                if i < self.max_links: # Only consider links within our padded array size
                    if link.get('index') == link_idx:
                        link_internal_idx = i
                        break
                else:
                    break # Stop searching if we exceed max_links

            if link_internal_idx == -1:
                 adv_logger.warning(f"Action link index {link_idx} not found within first {self.max_links} links of state. Storing with index 0.")
                 link_internal_idx = 0 # Fallback to index 0 if not found

            # Find the index of the new state in our link_states list
            action_idx = self.link_states.index(new_state) if new_state in self.link_states else 0

            # Create experience tuple (ensure numpy arrays are used for features)
            experience = (
                link_features,          # Numpy array
                signal_state,           # String
                (link_internal_idx, action_idx), # Tuple (int, int)
                float(reward),          # Ensure reward is float
                next_link_features,     # Numpy array
                next_signal_state,      # String
                bool(done)              # Ensure done is bool
            )

            if self.prioritized_replay:
                # Calculate initial priority (TD-error is better, but need to compute it first)
                # For now, add with max priority to ensure it gets sampled
                max_priority = 1.0
                if self.memory:
                    try:
                        # Find max priority among existing tuples (priority is the second element)
                        max_priority = max(p for _, p in self.memory)
                    except ValueError: # Handles case where memory might be temporarily empty during processing
                         max_priority = 1.0

                # Add to memory
                self.memory.append((experience, max_priority))

                # Keep memory within size limit - remove lowest priority if full? (More complex)
                # Simple approach: remove oldest
                if len(self.memory) > self.memory_size:
                    self.memory.pop(0)
            else:
                # Standard experience replay
                self.memory.append(experience) # deque handles maxlen automatically

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


        # --- Target Q-values with Double DQN ---
        with torch.no_grad():
            # 1. Get actions selected by the *online* model for the *next* state
            next_link_scores, next_action_values_online, _ = self.model(
                next_link_features_batch, next_signal_state_batch
            )
            # Find best link based on online model's scores
            # Need to consider only valid links if padding was used - find num actual links in next state if possible
            # Assuming for now next_link_scores are valid for comparison across links
            num_links_next = next_link_scores.size(1)
            best_link_indices_online = torch.argmax(next_link_scores, dim=1) # Shape: [batch_size]

            # Find best action *for that link* using online model's action values
            best_link_indices_online_exp = best_link_indices_online.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, next_action_values_online.size(2))
            q_values_best_link_online = torch.gather(next_action_values_online, 1, best_link_indices_online_exp.to(torch.int64)).squeeze(1) # Shape: [batch_size, num_actions]
            best_actions_online = torch.argmax(q_values_best_link_online, dim=1) # Shape: [batch_size]

            # 2. Evaluate these selected actions using the *target* model
            _, next_action_values_target, _ = self.target_model(
                next_link_features_batch, next_signal_state_batch
            )

            # Gather Q-values from target network for the best link selected by online network
            best_link_indices_online_exp2 = best_link_indices_online.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, next_action_values_target.size(2))
            q_values_best_link_target = torch.gather(next_action_values_target, 1, best_link_indices_online_exp2.to(torch.int64)).squeeze(1) # Shape: [batch_size, num_actions]

            # Gather the target network's Q-value for the action chosen by the online network
            # best_actions_online shape: [batch_size] -> unsqueeze to [batch_size, 1]
            q_target_next = torch.gather(q_values_best_link_target, 1, best_actions_online.unsqueeze(1)).squeeze(1) # Shape: [batch_size]

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

    def calculate_reward(self, state, action, next_state):
        """Calculate advanced reward with sophisticated metrics.
        # ... (rest of implementation - assumes state/action are valid) ...
        """
        if action is None or not isinstance(state, dict) or not isinstance(next_state, dict):
            return 0.0, {}

        link_index, new_state = action

        # Get metrics for all links from next_state
        next_link_states = next_state.get('link_states', [])
        if not next_link_states: return 0.0, {}

        # Find the specific link that was changed
        target_link = None
        for link in next_link_states:
            if link.get('index') == link_index:
                target_link = link
                break
        if not target_link: return 0.0, {}

        # Metrics for the targeted link
        target_waiting_time = target_link.get('waiting_time', 0.0)
        target_queue_length = target_link.get('queue_length', 0)

        # Overall metrics across all links
        total_waiting_time = sum(link.get('waiting_time', 0.0) for link in next_link_states)
        max_queue_length = max((link.get('queue_length', 0) for link in next_link_states), default=0)
        total_throughput = self.network.get_departed_vehicles_count() if hasattr(self.network, 'get_departed_vehicles_count') else 0

        # Flow smoothness (variance in queue lengths)
        queue_lengths = [link.get('queue_length', 0) for link in next_link_states]
        queue_balance = 0.0
        if len(queue_lengths) > 1:
            queue_variance = np.var(queue_lengths)
            queue_balance = -min(1.0, queue_variance / 100.0) # Penalty for high variance

        # Waiting time reduction
        norm_waiting_change = 0.0
        if self.last_state is not None and isinstance(self.last_state, dict):
            prev_link_states = self.last_state.get('link_states', [])
            prev_target_link = None
            for link in prev_link_states:
                if link.get('index') == link_index:
                    prev_target_link = link
                    break
            if prev_target_link:
                waiting_time_change = prev_target_link.get('waiting_time', 0.0) - target_waiting_time
                norm_waiting_change = max(-1.0, min(1.0, waiting_time_change / 100.0))

        # Signal state pattern efficiency
        signal_efficiency = 0.0
        current_signal_state = next_state.get('current_signal_state', '')
        for link in next_link_states:
            idx = link.get('index', -1)
            if 0 <= idx < len(current_signal_state):
                is_green = current_signal_state[idx] in 'Gg'
                has_traffic = link.get('vehicle_count', 0) > 0
                if (is_green and has_traffic) or (not is_green and not has_traffic):
                    signal_efficiency += 0.05
                elif is_green and not has_traffic:
                    signal_efficiency -= 0.05

        # Normalization constants
        MAX_WAITING_TIME = 1000.0
        MAX_QUEUE_LENGTH = 30.0
        MAX_THROUGHPUT = 30.0
        num_links = len(next_link_states) if next_link_states else 1

        # Normalize metrics safely
        norm_target_waiting = min(1.0, target_waiting_time / MAX_WAITING_TIME) if MAX_WAITING_TIME > 0 else 0.0
        norm_total_waiting = min(1.0, total_waiting_time / (MAX_WAITING_TIME * num_links)) if MAX_WAITING_TIME > 0 else 0.0
        norm_max_queue = min(1.0, max_queue_length / MAX_QUEUE_LENGTH) if MAX_QUEUE_LENGTH > 0 else 0.0
        norm_throughput = min(1.0, total_throughput / MAX_THROUGHPUT) if MAX_THROUGHPUT > 0 else 0.0

        # Component weights
        W_TARGET_WAITING = 0.25
        W_TOTAL_WAITING = 0.25
        W_THROUGHPUT = 0.15
        W_MAX_QUEUE = 0.05
        W_QUEUE_BALANCE = 0.1
        W_WAITING_CHANGE = 0.1
        W_SIGNAL_EFFICIENCY = 0.1

        # Calculate components
        target_waiting_component = -norm_target_waiting * W_TARGET_WAITING
        total_waiting_component = -norm_total_waiting * W_TOTAL_WAITING
        throughput_component = norm_throughput * W_THROUGHPUT
        max_queue_component = -norm_max_queue * W_MAX_QUEUE
        queue_balance_component = queue_balance * W_QUEUE_BALANCE
        waiting_change_component = norm_waiting_change * W_WAITING_CHANGE
        signal_efficiency_component = signal_efficiency * W_SIGNAL_EFFICIENCY

        # Combine components
        total_reward = (
            target_waiting_component + total_waiting_component +
            throughput_component + max_queue_component +
            queue_balance_component + waiting_change_component +
            signal_efficiency_component
        )

        components = {
            'target_waiting_comp': target_waiting_component,
            'total_waiting_comp': total_waiting_component,
            'throughput_comp': throughput_component,
            'max_queue_comp': max_queue_component,
            'queue_balance_comp': queue_balance_component,
            'waiting_change_comp': waiting_change_component,
            'signal_efficiency_comp': signal_efficiency_component
        }

        return total_reward, components


    # --- save_state and load_state methods ---
    def save_state(self, directory_path: str):
        """Saves the AdvancedAgent's state to the specified directory."""
        super().save_state(directory_path) # Creates directory via base class call
        adv_logger.info(f"Saving AdvancedAgent state for {self.tls_id} to {directory_path}")

        model_path = os.path.join(directory_path, 'model.pth')
        target_model_path = os.path.join(directory_path, 'target_model.pth')
        optimizer_path = os.path.join(directory_path, 'optimizer.pth')
        hyperparams_path = os.path.join(directory_path, 'hyperparams.json')

        try:
            # Save models and optimizer
            torch.save(self.model.state_dict(), model_path)
            torch.save(self.target_model.state_dict(), target_model_path)
            torch.save(self.optimizer.state_dict(), optimizer_path)

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


            adv_logger.info(f"AdvancedAgent state saved successfully for {self.tls_id}.")

        except Exception as e:
            adv_logger.error(f"Error saving AdvancedAgent state for {self.tls_id}: {e}", exc_info=True)


    def load_state(self, directory_path: str):
        """Loads the AdvancedAgent's state from the specified directory."""
        adv_logger.info(f"Loading AdvancedAgent state for {self.tls_id} from {directory_path}")

        model_path = os.path.join(directory_path, 'model.pth')
        target_model_path = os.path.join(directory_path, 'target_model.pth')
        optimizer_path = os.path.join(directory_path, 'optimizer.pth')
        hyperparams_path = os.path.join(directory_path, 'hyperparams.json')
        # Optional: Paths for memory, history
        # memory_path = os.path.join(directory_path, 'memory.pkl')
        # history_path = os.path.join(directory_path, 'history.pkl')

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

                # Note: memory_size, double_dqn, prioritized_replay are typically part of config,
                # loading them might change agent behavior significantly if config differs. Usually rely on __init__.

            # Set model modes correctly after loading
            self.model.train()
            self.target_model.eval()

            adv_logger.info(f"AdvancedAgent state loaded successfully for {self.tls_id}.")

        except FileNotFoundError:
            adv_logger.error(f"Error loading AdvancedAgent state: File not found during load attempt in {directory_path}")
        except Exception as e:
            adv_logger.error(f"Error loading AdvancedAgent state for {self.tls_id}: {e}", exc_info=True)

# --- END OF FILE advanced_agent.py ---