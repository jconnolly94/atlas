import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, defaultdict
from typing import Dict, Any, Tuple, Optional, List, Union

from .agent import Agent


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
        
        for state_str in state_strs:
            indices = self._state_to_indices(state_str)
            char_embeddings = self.char_embedding(indices)  # [seq_len, embedding_dim]
            
            # Apply attention
            attention_scores = self.attention(char_embeddings)  # [seq_len, 1]
            attention_weights = F.softmax(attention_scores, dim=0)  # [seq_len, 1]
            
            # Weight character embeddings by attention
            weighted_embeddings = char_embeddings * attention_weights  # [seq_len, embedding_dim]
            batch_embeddings.append(weighted_embeddings)
        
        # Stack into a batch
        return torch.stack(batch_embeddings)  # [batch_size, seq_len, embedding_dim]


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
        
        # Ensure we have at least one sample to process
        if batch_size == 0 or num_links == 0:
            # Return empty tensors of appropriate shapes if no data
            return (torch.zeros(0, num_links), 
                    torch.zeros(0, num_links, 2),  # 2 for G and r
                    torch.zeros(0, num_links, 8))  # 8 trend features
        
        # Encode link features
        link_embeddings = self.link_encoder(link_features)  # [batch_size, num_links, link_embedding_dim]
        
        try:
            # Encode signal states with attention
            signal_embeddings = self.signal_encoder(signal_state)  # [batch_size, max_signal_len, signal_embedding_dim]
            
            # Check signal embeddings shape for safety
            if signal_embeddings.dim() < 3:
                # Handle unexpected dimensions by creating a properly sized tensor of zeros
                signal_embeddings = torch.zeros(
                    batch_size, 
                    min(num_links, len(signal_state[0]) if signal_state and signal_state[0] else 0), 
                    self.signal_embedding_dim
                )
        except Exception as e:
            # Handle any unexpected errors during signal encoding
            print(f"Warning: Error encoding signal states: {e}")
            # Create a default tensor of zeros
            signal_embeddings = torch.zeros(batch_size, num_links, self.signal_embedding_dim)
        
        # Concatenate link embeddings with corresponding signal state embeddings
        combined_features = []
        for b in range(batch_size):
            # For each link, concatenate its embedding with the corresponding signal state embedding
            for l in range(num_links):
                # Make sure we don't exceed signal embedding dimensions
                if signal_embeddings.size(0) > b and signal_embeddings.size(1) > l:
                    combined = torch.cat([
                        link_embeddings[b, l],
                        signal_embeddings[b, l]
                    ])
                else:
                    # Pad with zeros if signal embedding is not available
                    signal_padding = torch.zeros(self.signal_embedding_dim)
                    combined = torch.cat([link_embeddings[b, l], signal_padding])
                combined_features.append(combined)
        
        # Reshape for processing
        combined_features = torch.stack(combined_features).view(
            batch_size, num_links, self.link_embedding_dim + self.signal_embedding_dim
        )
        
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
        "prioritized_replay": True  # Use prioritized experience replay
    }

    def __init__(self, tls_id, network, alpha=0.0005, gamma=0.98, epsilon=0.9,
                 epsilon_decay=0.9997, epsilon_min=0.2, batch_size=64,
                 memory_size=50000, target_update_freq=250, double_dqn=True,
                 prioritized_replay=True):
        """Initialize AdvancedAgent with lane-level control capabilities.

        Args:
            tls_id: ID of the traffic light this agent controls
            network: Network object providing access to simulation data
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_decay: Rate at which epsilon decreases
            epsilon_min: Minimum exploration rate
            batch_size: Batch size for learning
            memory_size: Size of experience replay memory
            target_update_freq: Frequency of target network updates
            double_dqn: Whether to use double DQN algorithm
            prioritized_replay: Whether to use prioritized experience replay
        """
        super().__init__(tls_id, network)
        self.network = network

        # Maximum number of links and features per link
        self.max_links = 24  # Support more links than standard DQN
        self.link_dim = 6    # Add an extra feature for historical trend
        
        # Possible link states (G for green, r for red)
        self.link_states = ['G', 'r']  # Simplified to just green and red
        
        # DQN hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.double_dqn = double_dqn
        self.prioritized_replay = prioritized_replay

        # Experience replay memory with prioritization if enabled
        if self.prioritized_replay:
            # Prioritized replay with (sample, priority) tuples
            self.memory = []
            self.memory_max_size = memory_size
            self.priority_alpha = 0.6  # Priority exponent
            self.priority_beta = 0.4   # Importance sampling exponent
            self.priority_beta_increment = 0.001  # Beta increment per update
            self.priority_epsilon = 0.01  # Small constant to avoid zero priority
        else:
            # Standard replay memory
            self.memory = deque(maxlen=memory_size)

        # Create main and target networks with advanced architecture
        self.model = AdvancedTrafficNetworkDQN(self.max_links, self.link_dim, len(self.link_states))
        self.target_model = AdvancedTrafficNetworkDQN(self.max_links, self.link_dim, len(self.link_states))

        # Copy weights from main to target network
        self.target_model.load_state_dict(self.model.state_dict())

        # Optimizer with weight decay for regularization
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha, weight_decay=1e-5)

        # Target network update frequency
        self.target_update_freq = target_update_freq
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
        
        Args:
            tls_id: ID of the traffic light this agent controls
            network: Network object providing access to simulation data
            **kwargs: Additional configuration parameters
            
        Returns:
            Properly configured AdvancedAgent instance
        """
        # Start with default configuration
        config = cls.DEFAULT_CONFIG.copy()
        
        # Override with provided kwargs
        config.update(kwargs)
        
        # Create and return instance
        return cls(tls_id, network, **config)

    def _preprocess_state(self, state):
        """Convert environment state dict to tensors for the neural network.
        
        Args:
            state: State dictionary from environment
            
        Returns:
            Tuple of (link_features, signal_state)
        """
        # Extract link states and current signal state
        link_states = state['link_states']
        current_signal_state = state['current_signal_state']
        
        # Convert link states to a tensor with additional features
        # Features: [queue_length, waiting_time, vehicle_count, time_since_last_change, is_green, historical_trend]
        link_features = []
        
        for link in link_states:
            link_index = link['index']
            # Check if current state is green (G or g)
            is_green = 1.0 if current_signal_state[link_index] in 'Gg' else 0.0
            
            # Calculate historical trend (average queue growth over recent history)
            historical_trend = 0.0
            if link_index in self.traffic_history and len(self.traffic_history[link_index]) > 1:
                queue_history = [record['queue_length'] for record in self.traffic_history[link_index]]
                if len(queue_history) >= 2:
                    # Calculate average rate of change
                    changes = [(queue_history[i] - queue_history[i-1]) for i in range(1, len(queue_history))]
                    historical_trend = sum(changes) / len(changes)
                    # Normalize to range [-1, 1]
                    historical_trend = max(-1.0, min(1.0, historical_trend / 5.0))
            
            features = [
                link['queue_length'] / 30.0,  # Normalize queue length (higher max than standard)
                link['waiting_time'] / 1000.0,  # Normalize waiting time (higher max than standard)
                link['vehicle_count'] / 30.0,  # Normalize vehicle count
                link['time_since_last_change'] / 200.0,  # Normalize time since last change
                is_green,
                historical_trend  # Historical trend feature
            ]
            link_features.append(features)
            
            # Update traffic history for this link
            self.traffic_history[link_index].append({
                'queue_length': link['queue_length'],
                'waiting_time': link['waiting_time'],
                'vehicle_count': link['vehicle_count']
            })
            
        # Pad to max_links if necessary
        while len(link_features) < self.max_links:
            link_features.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
        # Truncate if too many links
        link_features = link_features[:self.max_links]
        
        return np.array(link_features), current_signal_state

    def choose_action(self, state):
        """Choose an action based on the current state with advanced exploration strategies.
        
        Args:
            state: Current state dictionary with link_states and current_signal_state
            
        Returns:
            Tuple of (link_index, new_state) or None if no action
        """
        # Store state for later use
        self.last_state = state
        
        # Handle invalid state format
        if not isinstance(state, dict) or 'link_states' not in state:
            print(f"Warning: Advanced agent received incompatible state format. Expected dict with 'link_states' key.")
            return None
            
        # Get link states and validate
        link_states = state['link_states']
        if not link_states:
            return None  # No links to control
            
        # Preprocess state for neural network
        link_features, signal_state = self._preprocess_state(state)
        
        # Adaptive exploration strategy based on recent performance
        if len(self.performance_history) > 0:
            # Calculate the average of recent rewards
            recent_rewards = list(self.performance_history)  # Convert deque to list
            recent_rewards_subset = recent_rewards[-min(5, len(recent_rewards)):]  # Take last 5 or fewer
            avg_reward = sum(recent_rewards_subset) / len(recent_rewards_subset)
            
            # If rewards are consistently poor, increase exploration
            if avg_reward < -0.5:
                exploration_rate = min(1.0, self.epsilon * 1.2)
            else:
                exploration_rate = self.epsilon
        else:
            exploration_rate = self.epsilon
        
        # Exploration: choose random action with probability epsilon
        if np.random.rand() <= exploration_rate:
            # Advanced exploration strategy: weighted sampling based on metrics
            
            # Create weights combining queue length, waiting time, and trend
            weights = []
            for link in link_states:
                queue_weight = link['queue_length'] + 0.1  # Avoid zero weight
                waiting_weight = link['waiting_time'] / 100.0 + 0.1  # Scale waiting time
                
                # Get historical trend from traffic history
                link_index = link['index']
                trend_weight = 0.1  # Default weight
                if link_index in self.traffic_history and len(self.traffic_history[link_index]) > 1:
                    queue_history = [record['queue_length'] for record in self.traffic_history[link_index]]
                    if len(queue_history) >= 2:
                        # Calculate recent trend (positive = growing queue, negative = shrinking)
                        recent_trend = queue_history[-1] - queue_history[0]
                        # Growing queues get higher weight
                        trend_weight = max(0.1, recent_trend / 5.0 + 0.1)
                
                # Combine weights with different emphasis
                combined_weight = (queue_weight * 0.5) + (waiting_weight * 0.4) + (trend_weight * 0.1)
                weights.append(combined_weight)
            
            # Normalize weights to probabilities
            total_weight = sum(weights)
            if total_weight > 0:
                probs = [w/total_weight for w in weights]
                link_idx = np.random.choice(len(link_states), p=probs)
            else:
                link_idx = np.random.choice(len(link_states))
                
            # Get the current state of this link
            link_index = link_states[link_idx]['index']
            current_link_state = state['current_signal_state'][link_index]
            
            # Choose a new state different from the current one
            if current_link_state in 'Gg':
                new_state = 'r'  # Change green to red
            else:
                new_state = 'G'  # Change red/yellow to green
                
            action = (link_index, new_state)
        else:
            # Exploitation: Use the advanced DQN to select the best action
            
            # Convert to torch tensor and add batch dimension
            link_tensor = torch.FloatTensor(link_features).unsqueeze(0)
            
            with torch.no_grad():
                # Get link scores and action values from advanced model
                link_scores, action_values, trend_features = self.model(link_tensor, [signal_state])
                
                # Use trend features to adjust link scores
                valid_links = min(len(link_states), self.max_links)
                
                # Calculate priority scores that combine immediate value and trend
                priority_scores = link_scores[0, :valid_links].clone()
                
                # Adjust scores based on trend features if available
                if trend_features.size(0) > 0 and trend_features.size(1) >= valid_links:
                    for i in range(valid_links):
                        # Extract trend insights (e.g. if queue is growing, increase priority)
                        trend_insight = trend_features[0, i].mean().item()
                        # Scale the adjustment based on trend strength (-0.2 to +0.2)
                        adjustment = trend_insight * 0.2
                        priority_scores[i] += adjustment
                
                # Get the best link to change based on adjusted scores
                best_link_idx = priority_scores.argmax().item()
                
                # Get the best action for this link
                best_action_idx = action_values[0, best_link_idx].argmax().item()
                
                # Map to actual link index and state
                link_index = link_states[best_link_idx]['index']
                new_state = self.link_states[best_action_idx]
                
                # Advanced policy: Don't change if the link is already in the desired state
                # and there's no significant queue or waiting time
                current_link_state = state['current_signal_state'][link_index]
                current_queue = link_states[best_link_idx]['queue_length']
                current_waiting = link_states[best_link_idx]['waiting_time']
                
                if ((new_state == 'G' and current_link_state in 'Gg') or 
                    (new_state == 'r' and current_link_state in 'Rr')) and \
                   current_queue < 3 and current_waiting < 100:
                    # Find another high-priority link
                    temp_scores = priority_scores.clone()
                    temp_scores[best_link_idx] = float('-inf')
                    second_best_link_idx = temp_scores.argmax().item()
                    
                    # Get the best action for this link
                    second_best_action_idx = action_values[0, second_best_link_idx].argmax().item()
                    
                    # Map to actual link index and state
                    link_index = link_states[second_best_link_idx]['index']
                    new_state = self.link_states[second_best_action_idx]
                
            action = (link_index, new_state)
        
        # Update epsilon with adaptive decay
        # If average reward is improving, slow down decay to stabilize
        if len(self.performance_history) >= 10:
            # Calculate recent and older averages from list instead of using slices
            recent_rewards = list(self.performance_history)
            recent_avg = sum(recent_rewards[-5:]) / min(5, len(recent_rewards))
            older_avg = sum(recent_rewards[:-5]) / max(1, len(recent_rewards) - 5)
            
            if recent_avg > older_avg:
                # Rewards are improving, slow down decay
                effective_decay = self.epsilon_decay * 1.001  # Slower decay
            else:
                # Rewards are not improving, speed up decay slightly
                effective_decay = self.epsilon_decay * 0.999  # Faster decay
                
            self.epsilon = max(self.epsilon_min, self.epsilon * effective_decay)
        else:
            # Standard decay if not enough history
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Store last action for learning
        self.last_action = action
        return action

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory with prioritization.

        Args:
            state: Current state dictionary
            action: Action taken as (link_index, new_state)
            reward: Reward received
            next_state: Next state dictionary
            done: Whether episode is done
        """
        if action is None:
            return  # Skip storing if no action was taken
            
        try:    
            # Preprocess states
            link_features, signal_state = self._preprocess_state(state)
            next_link_features, next_signal_state = self._preprocess_state(next_state)
            
            # Extract link index and new state
            link_idx, new_state = action
            
            # Find the internal index of the link in our link_states list
            link_internal_idx = None
            for i, link in enumerate(state['link_states']):
                if link['index'] == link_idx:
                    link_internal_idx = i
                    break
                    
            if link_internal_idx is None:
                # If the link index isn't found, map to the first link as fallback
                link_internal_idx = 0
            
            # Safety check on link_internal_idx bounds
            link_internal_idx = min(link_internal_idx, self.max_links - 1)
                
            # Find the index of the new state in our link_states list
            action_idx = self.link_states.index(new_state) if new_state in self.link_states else 0
            
            # Create experience tuple
            experience = (
                link_features, 
                signal_state,
                (link_internal_idx, action_idx),
                reward, 
                next_link_features,
                next_signal_state,
                done
            )
            
            if self.prioritized_replay:
                # For first experiences, use max priority
                max_priority = max([p for _, p in self.memory]) if self.memory else 1.0
                
                # Add to memory with high initial priority to ensure sampling
                self.memory.append((experience, max_priority))
                
                # Keep memory within size limit
                if len(self.memory) > self.memory_max_size:
                    self.memory.pop(0)  # Remove oldest experience
            else:
                # Standard experience replay
                self.memory.append(experience)
                
        except Exception as e:
            # Log any errors but don't crash
            print(f"Error remembering experience: {e}")

    def learn(self, state, action, next_state, done):
        """Learn from experience using advanced deep Q-learning.

        Args:
            state: Previous state dictionary
            action: Action that was taken as (link_index, new_state)
            next_state: Resulting state dictionary
            done: Whether episode is done
        """
        # Skip learning if no action was taken
        if action is None:
            return
            
        # Calculate reward
        reward, _ = self.calculate_reward(state, action, next_state)
        
        # Store reward in performance history
        self.performance_history.append(reward)

        # Store experience
        self.remember(state, action, reward, next_state, done)

        # Only learn if we have enough samples
        if (self.prioritized_replay and len(self.memory) < self.batch_size) or \
           (not self.prioritized_replay and len(self.memory) < self.batch_size):
            return

        # Perform experience replay
        self._replay()

    def _replay(self):
        """Perform experience replay with advanced techniques."""
        # Sample from memory
        if self.prioritized_replay:
            # Sample based on priorities
            priorities = np.array([p for _, p in self.memory])
            # Apply priority exponent
            priorities = priorities ** self.priority_alpha
            # Calculate sampling probabilities
            probs = priorities / priorities.sum()
            
            # Sample indices
            indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
            
            # Extract experiences and priorities
            batch = [self.memory[idx][0] for idx in indices]
            batch_priorities = np.array([self.memory[idx][1] for idx in indices])
            
            # Calculate importance sampling weights
            weights = (len(self.memory) * probs[indices]) ** (-self.priority_beta)
            # Normalize weights
            weights = weights / weights.max()
            # Convert to tensor
            importance_weights = torch.FloatTensor(weights)
            
            # Increment beta
            self.priority_beta = min(1.0, self.priority_beta + self.priority_beta_increment)
        else:
            # Standard uniform sampling
            batch = random.sample(self.memory, self.batch_size)
            importance_weights = torch.ones(self.batch_size)  # All weights equal
            indices = None  # Not used for standard replay

        # Extract batch components
        link_features_batch = np.array([t[0] for t in batch])
        signal_state_batch = [t[1] for t in batch]
        actions_batch = [t[2] for t in batch]
        rewards_batch = np.array([t[3] for t in batch])
        next_link_features_batch = np.array([t[4] for t in batch])
        next_signal_state_batch = [t[5] for t in batch]
        dones_batch = np.array([t[6] for t in batch])
        
        # Convert to tensors
        link_features_tensor = torch.FloatTensor(link_features_batch)
        rewards_tensor = torch.FloatTensor(rewards_batch)
        next_link_features_tensor = torch.FloatTensor(next_link_features_batch)
        dones_tensor = torch.FloatTensor(dones_batch)
        
        # Separate link indices and action indices
        link_indices = torch.LongTensor([a[0] for a in actions_batch])
        action_indices = torch.LongTensor([a[1] for a in actions_batch])
        
        # Current Q-values
        link_scores, action_values, _ = self.model(link_features_tensor, signal_state_batch)
        
        # Extract Q-values for the selected links and actions
        # First index by batch, then by link, then by action
        current_q = torch.zeros(self.batch_size)
        for i in range(self.batch_size):
            link_idx = link_indices[i]
            if link_idx >= action_values.size(1):
                # Safety check - if link_idx is out of bounds, use index 0
                link_idx = 0
            action_idx = action_indices[i]
            if action_idx >= action_values.size(2):
                # Safety check - if action_idx is out of bounds, use index 0
                action_idx = 0
            current_q[i] = action_values[i, link_idx, action_idx]
            
        # Target Q-values with Double DQN approach
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use online network to select action, target network to evaluate
                next_link_scores, next_action_values, _ = self.model(
                    next_link_features_tensor, next_signal_state_batch
                )
                
                # Target network values
                _, target_action_values, _ = self.target_model(
                    next_link_features_tensor, next_signal_state_batch
                )
                
                target_q = torch.zeros(self.batch_size)
                for i in range(self.batch_size):
                    # Get best link according to online network
                    valid_links = min(next_link_scores.size(1), self.max_links)
                    if valid_links > 0:
                        best_link_idx = next_link_scores[i, :valid_links].argmax().item()
                        
                        # Get best action for that link from online network
                        if best_link_idx < next_action_values.size(1):
                            best_action_idx = next_action_values[i, best_link_idx].argmax().item()
                            
                            # Get Q-value from target network for that link+action
                            if best_link_idx < target_action_values.size(1) and \
                               best_action_idx < target_action_values.size(2):
                                best_action_val = target_action_values[i, best_link_idx, best_action_idx].item()
                            else:
                                best_action_val = 0.0
                        else:
                            best_action_val = 0.0
                    else:
                        best_action_val = 0.0
                    
                    # Calculate target Q-value
                    target_q[i] = rewards_tensor[i] + (1 - dones_tensor[i]) * self.gamma * best_action_val
            else:
                # Standard DQN: use target network to select and evaluate
                next_link_scores, next_action_values, _ = self.target_model(
                    next_link_features_tensor, next_signal_state_batch
                )
                
                target_q = torch.zeros(self.batch_size)
                for i in range(self.batch_size):
                    # Get best link and action from target network
                    valid_links = min(next_link_scores.size(1), self.max_links)
                    if valid_links > 0:
                        best_link_idx = next_link_scores[i, :valid_links].argmax().item()
                        
                        if best_link_idx < next_action_values.size(1):
                            best_action_val = next_action_values[i, best_link_idx].max().item()
                        else:
                            best_action_val = 0.0
                    else:
                        best_action_val = 0.0
                    
                    # Calculate target Q-value
                    target_q[i] = rewards_tensor[i] + (1 - dones_tensor[i]) * self.gamma * best_action_val
                
        # Compute loss with importance sampling for prioritized replay
        losses = F.mse_loss(current_q, target_q, reduction='none')
        loss = (losses * importance_weights).mean()
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities for prioritized replay
        if self.prioritized_replay and indices is not None:
            # Get new priorities based on TD error
            td_errors = torch.abs(target_q - current_q).detach().cpu().numpy()
            
            # Update priorities in memory
            for idx, error in zip(indices, td_errors):
                # Add small constant to avoid zero priority
                new_priority = float(error) + self.priority_epsilon
                self.memory[idx] = (self.memory[idx][0], new_priority)

        # Update target network if needed
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def calculate_reward(self, state, action, next_state):
        """Calculate advanced reward with sophisticated metrics.

        This implementation incorporates traffic flow smoothness,
        prioritizes links with longer wait times, and rewards
        coordinated signal patterns.

        Args:
            state: Previous state
            action: Action taken as (link_index, new_state)
            next_state: Resulting state

        Returns:
            reward: Calculated reward value
            components: Dictionary of reward components
        """
        if action is None:
            return 0.0, {}  # No action, no reward
            
        link_index, new_state = action
        
        # Get metrics for all links
        next_link_states = next_state['link_states']
        
        # Find the specific link that was changed
        target_link = None
        for link in next_link_states:
            if link['index'] == link_index:
                target_link = link
                break
                
        if not target_link:
            return 0.0, {}  # Link not found
        
        # Metrics for the targeted link
        target_waiting_time = target_link['waiting_time']
        target_queue_length = target_link['queue_length']
        
        # Overall metrics across all links
        total_waiting_time = sum(link['waiting_time'] for link in next_link_states)
        max_queue_length = max((link['queue_length'] for link in next_link_states), default=0)
        total_throughput = self.network.get_departed_vehicles_count()
        
        # Advanced metric: Flow smoothness (variance in queue lengths)
        queue_lengths = [link['queue_length'] for link in next_link_states]
        if len(queue_lengths) > 1:
            queue_variance = np.var(queue_lengths)
            # Lower variance is better (more balanced queues)
            queue_balance = -min(1.0, queue_variance / 100.0)
        else:
            queue_balance = 0.0
            
        # Advanced metric: Waiting time reduction
        if self.last_state is not None:
            # Find previous state of target link
            prev_target_link = None
            for link in self.last_state['link_states']:
                if link['index'] == link_index:
                    prev_target_link = link
                    break
                    
            if prev_target_link:
                # Calculate waiting time change
                waiting_time_change = prev_target_link['waiting_time'] - target_waiting_time
                # Normalize to range [-1, 1]
                norm_waiting_change = max(-1.0, min(1.0, waiting_time_change / 100.0))
            else:
                norm_waiting_change = 0.0
        else:
            norm_waiting_change = 0.0
            
        # Advanced metric: Signal state pattern efficiency
        # Reward for having green light on links with traffic, red on empty links
        signal_efficiency = 0.0
        current_signal_state = next_state['current_signal_state']
        
        for link in next_link_states:
            idx = link['index']
            if idx < len(current_signal_state):
                is_green = current_signal_state[idx] in 'Gg'
                has_traffic = link['vehicle_count'] > 0
                
                if (is_green and has_traffic) or (not is_green and not has_traffic):
                    # Good state: green for links with traffic, red for empty links
                    signal_efficiency += 0.05
                elif is_green and not has_traffic:
                    # Wasted green: penalty
                    signal_efficiency -= 0.05
                    
        # Normalization constants (adjusted for more extreme values)
        MAX_WAITING_TIME = 1000.0
        MAX_QUEUE_LENGTH = 30.0
        MAX_THROUGHPUT = 30.0
        
        # Normalize metrics
        norm_target_waiting = min(1.0, target_waiting_time / MAX_WAITING_TIME)
        norm_total_waiting = min(1.0, total_waiting_time / (MAX_WAITING_TIME * len(next_link_states)))
        norm_max_queue = min(1.0, max_queue_length / MAX_QUEUE_LENGTH)
        norm_throughput = min(1.0, total_throughput / MAX_THROUGHPUT)
        
        # Component weights with more emphasis on improvements and smoothness
        W_TARGET_WAITING = 0.25    # Waiting time for target link
        W_TOTAL_WAITING = 0.25     # Total waiting time across all links
        W_THROUGHPUT = 0.15        # Overall throughput
        W_MAX_QUEUE = 0.05         # Maximum queue length (prevent extremes)
        W_QUEUE_BALANCE = 0.1      # Queue balance (smoothness of flow)
        W_WAITING_CHANGE = 0.1     # Improvement in waiting time
        W_SIGNAL_EFFICIENCY = 0.1  # Efficiency of signal pattern
        
        # Calculate components (negative waiting times = penalties)
        target_waiting_component = -norm_target_waiting * W_TARGET_WAITING
        total_waiting_component = -norm_total_waiting * W_TOTAL_WAITING
        throughput_component = norm_throughput * W_THROUGHPUT
        max_queue_component = -norm_max_queue * W_MAX_QUEUE
        queue_balance_component = queue_balance * W_QUEUE_BALANCE
        waiting_change_component = norm_waiting_change * W_WAITING_CHANGE
        signal_efficiency_component = signal_efficiency * W_SIGNAL_EFFICIENCY
        
        # Combine components
        total_reward = (
            target_waiting_component +
            total_waiting_component +
            throughput_component +
            max_queue_component +
            queue_balance_component +
            waiting_change_component +
            signal_efficiency_component
        )
        
        # Return reward and components for analysis
        components = {
            'target_waiting_component': target_waiting_component,
            'total_waiting_component': total_waiting_component,
            'throughput_component': throughput_component,
            'max_queue_component': max_queue_component,
            'queue_balance_component': queue_balance_component,
            'waiting_change_component': waiting_change_component,
            'signal_efficiency_component': signal_efficiency_component
        }
        
        return total_reward, components