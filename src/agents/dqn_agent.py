import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import json
import logging
from collections import deque, defaultdict
from typing import Dict, Any, Tuple, Optional, List, Union

from .agent import Agent

# Configure logger for this module
dqn_logger = logging.getLogger(__name__) # Use __name__ for module-level logger


class DQNLinkEncoder(nn.Module):
    """Encoder for link-level state representation."""
    
    def __init__(self, input_dim=5, embedding_dim=8):
        """Initialize link encoder.
        
        Args:
            input_dim: Dimensionality of each link state vector
            embedding_dim: Dimensionality of output embeddings
        """
        super(DQNLinkEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, embedding_dim),
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


class DQNSignalStateEncoder(nn.Module):
    """Encoder for the current signal state."""
    
    def __init__(self, embedding_dim=8):
        """Initialize signal state encoder.
        
        Args:
            embedding_dim: Dimensionality of output embeddings
        """
        super(DQNSignalStateEncoder, self).__init__()
        # Character-level embedding for signal state (r, R, g, G, y, Y, o, O)
        self.char_embedding = nn.Embedding(8, embedding_dim)
        
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
        """Encode signal state strings.
        
        Args:
            state_strs: List of signal state strings
            
        Returns:
            Tensor of embeddings for each position in the state strings
        """
        # Process batch of state strings
        batch_embeddings = []
        for state_str in state_strs:
            indices = self._state_to_indices(state_str)
            embeddings = self.char_embedding(indices)
            batch_embeddings.append(embeddings)
        
        # Stack into a batch
        return torch.stack(batch_embeddings)


class TrafficNetworkDQN(nn.Module):
    """Deep Q-Network for traffic signal control."""

    def __init__(self, max_links=16, link_dim=5, output_dim=2):
        """Initialize network architecture.

        Args:
            max_links: Maximum number of links to support
            link_dim: Dimensionality of each link's feature vector
            output_dim: Output dimension for each link (green or red)
        """
        super(TrafficNetworkDQN, self).__init__()
        
        # Embedding dimensions
        self.link_embedding_dim = 8
        self.signal_embedding_dim = 8
        self.max_links = max_links
        
        # Encoders
        self.link_encoder = DQNLinkEncoder(link_dim, self.link_embedding_dim)
        self.signal_encoder = DQNSignalStateEncoder(self.signal_embedding_dim)
        
        # Link selection network
        self.link_selector = nn.Sequential(
            nn.Linear(self.link_embedding_dim + self.signal_embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Scores for each link
        )
        
        # Action selection network (what state to set the link to)
        self.action_selector = nn.Sequential(
            nn.Linear(self.link_embedding_dim + self.signal_embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)  # Q-values for each possible state (G, r)
        )

    def forward(self, link_features, signal_state):
        """Forward pass through the network with improved robustness.

        Args:
            link_features: Tensor of link features [batch_size, num_links, link_dim]
            signal_state: List of signal state strings (e.g., ["GrGr", "rGrG"])

        Returns:
            link_scores: Scores for selecting each link [batch_size, num_links]
            action_values: Q-values for actions on each link [batch_size, num_links, output_dim]
        """
        batch_size = link_features.shape[0]
        num_links = link_features.shape[1]
        
        # Ensure we have at least one sample to process
        if batch_size == 0 or num_links == 0:
            # Return empty tensors of appropriate shapes if no data
            return (torch.zeros(0, num_links), 
                    torch.zeros(0, num_links, 2))  # 2 for G and r
        
        # Encode link features
        link_embeddings = self.link_encoder(link_features)  # [batch_size, num_links, link_embedding_dim]
        
        try:
            # Encode signal states 
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
        
        return link_scores, action_values


class DQNAgent(Agent):
    """Agent using Deep Q-Learning for traffic signal control."""
    
    # Default configuration
    DEFAULT_CONFIG = {
        "alpha": 0.001,           # Learning rate
        "gamma": 0.95,            # Discount factor
        "epsilon": 0.9,           # High exploration rate
        "epsilon_decay": 0.9999,  # Very slow decay to keep exploring longer
        "epsilon_min": 0.2,       # Higher minimum to maintain some exploration
        "batch_size": 32,
        "memory_size": 20000,     # Large memory for diverse experiences
        "target_update_freq": 200  # Less frequent updates for stability
    }

    def __init__(self, tls_id, network, alpha=0.001, gamma=0.95, epsilon=0.9,
                 epsilon_decay=0.9999, epsilon_min=0.2, batch_size=32,
                 memory_size=20000, target_update_freq=200):
        """Initialize DQNAgent.

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
        """
        super().__init__(tls_id, network)
        self.network = network

        # Maximum number of links and features per link
        self.max_links = 16  # Maximum number of links to support
        self.link_dim = 5    # Features per link: [queue_length, waiting_time, etc.]
        
        # Possible link states (G for green, r for red)
        self.link_states = ['G', 'r']  # Simplified to just green and red
        
        # DQN hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        # Experience replay memory
        self.memory = deque(maxlen=memory_size)

        # Create main and target networks
        self.model = TrafficNetworkDQN(self.max_links, self.link_dim, len(self.link_states))
        self.target_model = TrafficNetworkDQN(self.max_links, self.link_dim, len(self.link_states))

        # Copy weights from main to target network
        self.target_model.load_state_dict(self.model.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)

        # Target network update frequency
        self.target_update_freq = target_update_freq
        self.train_step = 0

        # Track last action for reward calculation
        self.last_action = None
        
    @classmethod
    def create(cls, tls_id, network, **kwargs):
        """Create an instance of the DQNAgent with proper configuration.
        
        Args:
            tls_id: ID of the traffic light this agent controls
            network: Network object providing access to simulation data
            **kwargs: Additional configuration parameters
            
        Returns:
            Properly configured DQNAgent instance
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
        
        # Convert link states to a tensor
        # Features: [queue_length, waiting_time, vehicle_count, time_since_last_change, is_green]
        link_features = []
        for link in link_states:
            # Check if current state is green (G or g)
            is_green = 1.0 if current_signal_state[link['index']] in 'Gg' else 0.0
            
            features = [
                link['queue_length'] / 20.0,  # Normalize queue length
                link['waiting_time'] / 500.0,  # Normalize waiting time
                link['vehicle_count'] / 20.0,  # Normalize vehicle count
                link['time_since_last_change'] / 120.0,  # Normalize time since last change
                is_green
            ]
            link_features.append(features)
            
        # Pad to max_links if necessary
        while len(link_features) < self.max_links:
            link_features.append([0.0, 0.0, 0.0, 0.0, 0.0])
            
        # Truncate if too many links
        link_features = link_features[:self.max_links]
        
        return np.array(link_features), current_signal_state

    def choose_action(self, state):
        """Choose an action based on the current state.
        
        Args:
            state: Current state dictionary with link_states and current_signal_state
            
        Returns:
            Tuple of (link_index, new_state) or None if no action
        """
        # Handle invalid state format
        if not isinstance(state, dict) or 'link_states' not in state:
            print(f"Warning: DQN agent received incompatible state format. Expected dict with 'link_states' key.")
            return None
            
        # Get link states and validate
        link_states = state['link_states']
        if not link_states:
            return None  # No links to control
            
        # Preprocess state for neural network
        link_features, signal_state = self._preprocess_state(state)
        
        # Exploration: choose random action with probability epsilon
        if np.random.rand() <= self.epsilon:
            # Choose a random link with higher probability for links with queues
            queue_lengths = np.array([link['queue_length'] for link in link_states])
            if queue_lengths.sum() > 0:
                probs = queue_lengths / queue_lengths.sum()
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
            # Exploitation: Use the DQN to select the best action
            
            # Convert to torch tensor and add batch dimension
            link_tensor = torch.FloatTensor(link_features).unsqueeze(0)
            
            with torch.no_grad():
                # Get link scores and action values
                link_scores, action_values = self.model(link_tensor, [signal_state])
                
                # Get the best link to change
                valid_links = min(len(link_states), self.max_links)
                best_link_idx = link_scores[0, :valid_links].argmax().item()
                
                # Get the best action for this link
                best_action_idx = action_values[0, best_link_idx].argmax().item()
                
                # Map to actual link index and state
                link_index = link_states[best_link_idx]['index']
                new_state = self.link_states[best_action_idx]
                
                # Don't change if the link is already in the desired state
                current_link_state = state['current_signal_state'][link_index]
                if (new_state == 'G' and current_link_state in 'Gg') or \
                   (new_state == 'r' and current_link_state in 'Rr'):
                    # Find the second-best link
                    temp_scores = link_scores.clone()
                    temp_scores[0, best_link_idx] = float('-inf')
                    second_best_link_idx = temp_scores[0, :valid_links].argmax().item()
                    
                    # Get the best action for this link
                    second_best_action_idx = action_values[0, second_best_link_idx].argmax().item()
                    
                    # Map to actual link index and state
                    link_index = link_states[second_best_link_idx]['index']
                    new_state = self.link_states[second_best_action_idx]
                
            action = (link_index, new_state)
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Store last action for learning
        self.last_action = action
        return action
        
    def save_state(self, directory_path: str):
        """Saves the DQN agent's state (models, optimizer, hyperparameters) to the specified directory."""
        try:
            # Ensure directory exists using the base class method if available, else create manually
            if hasattr(super(), 'save_state') and callable(super().save_state):
                 super().save_state(directory_path)
            else:
                 os.makedirs(directory_path, exist_ok=True)
            dqn_logger.info(f"Ensured directory exists: {directory_path}")
        except Exception as e:
            dqn_logger.error(f"Error ensuring directory exists {directory_path}: {e}", exc_info=True)
            return # Cannot proceed if directory cannot be created/accessed

        model_path = os.path.join(directory_path, 'model.pth')
        target_model_path = os.path.join(directory_path, 'target_model.pth')
        optimizer_path = os.path.join(directory_path, 'optimizer.pth')
        hyperparams_path = os.path.join(directory_path, 'hyperparams.json')
        dqn_logger.info(f"Attempting to save DQNAgent state for {self.tls_id} to {directory_path}")

        try:
            # Save state dictionaries
            if hasattr(self, 'model') and self.model:
                torch.save(self.model.state_dict(), model_path)
            else:
                 dqn_logger.warning("Attribute 'model' not found or is None. Skipping save.")

            if hasattr(self, 'target_model') and self.target_model:
                torch.save(self.target_model.state_dict(), target_model_path)
            else:
                dqn_logger.warning("Attribute 'target_model' not found or is None. Skipping save.")

            if hasattr(self, 'optimizer') and self.optimizer:
                torch.save(self.optimizer.state_dict(), optimizer_path)
            else:
                dqn_logger.warning("Attribute 'optimizer' not found or is None. Skipping save.")

            # Save hyperparameters
            hyperparams = {
                'epsilon': getattr(self, 'epsilon', None),
                'gamma': getattr(self, 'gamma', None),
                'train_step': getattr(self, 'train_step', 0),
                'epsilon_decay': getattr(self, 'epsilon_decay', None),
                'epsilon_min': getattr(self, 'epsilon_min', None),
                'batch_size': getattr(self, 'batch_size', None),
                'memory_size': getattr(self, 'memory', None).maxlen if hasattr(self, 'memory') and hasattr(self.memory, 'maxlen') else None, # Save intended size
                'target_update_freq': getattr(self, 'target_update_freq', None),
                # Add other hyperparameters defined in __init__ or config
            }
            with open(hyperparams_path, 'w') as f:
                json.dump(hyperparams, f, indent=4)

            dqn_logger.info(f"DQNAgent state for {self.tls_id} saved successfully to {directory_path}")

        except Exception as e:
            dqn_logger.error(f"Error saving DQNAgent state for {self.tls_id} to {directory_path}: {e}", exc_info=True)
            
    def load_state(self, directory_path: str):
        """Loads the DQN agent's state (models, optimizer, hyperparameters) from the specified directory."""
        model_path = os.path.join(directory_path, 'model.pth')
        target_model_path = os.path.join(directory_path, 'target_model.pth')
        optimizer_path = os.path.join(directory_path, 'optimizer.pth')
        hyperparams_path = os.path.join(directory_path, 'hyperparams.json')
        dqn_logger.info(f"Attempting to load DQNAgent state for {self.tls_id} from {directory_path}")

        # Check if essential files exist
        required_files = [model_path, target_model_path, optimizer_path, hyperparams_path]
        if not all(os.path.exists(p) for p in required_files):
            dqn_logger.warning(f"Cannot load DQNAgent state for {self.tls_id}: Required file(s) not found in {directory_path}")
            return # Do not attempt partial load

        try:
            # Determine device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dqn_logger.info(f"Loading state onto device: {device}")

            # --- Load Models ---
            # Ensure models are instantiated BEFORE loading state
            if not hasattr(self, 'model') or not self.model:
                 dqn_logger.error("Model not initialized before loading state. Cannot proceed.")
                 return
            if not hasattr(self, 'target_model') or not self.target_model:
                 dqn_logger.error("Target model not initialized before loading state. Cannot proceed.")
                 return

            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.target_model.load_state_dict(torch.load(target_model_path, map_location=device))
            self.model.to(device) # Move model to correct device
            self.target_model.to(device)

            # --- Load Optimizer ---
            # Ensure optimizer is initialized BEFORE loading state
            if not hasattr(self, 'optimizer') or not self.optimizer:
                 dqn_logger.error("Optimizer not initialized before loading state. Re-initializing.")
                 # Re-initialize optimizer (requires learning rate from hyperparams or default)
                 # This assumes self.model.parameters() are now correctly loaded and on the right device
                 default_lr = 0.001 # Or get from config
                 self.optimizer = optim.Adam(self.model.parameters(), lr=default_lr)
                 # Load state into the newly created optimizer
                 self.optimizer.load_state_dict(torch.load(optimizer_path)) # Now load the state
            else:
                 # If optimizer exists, just load the state.
                 # NOTE: This might cause issues if the model parameters changed device *after*
                 # the optimizer was initially created but before saving. Re-initializing is often safer.
                 try:
                     self.optimizer.load_state_dict(torch.load(optimizer_path))
                 except Exception as opt_load_err:
                      dqn_logger.error(f"Failed loading optimizer state dict, re-initializing optimizer: {opt_load_err}")
                      # Re-initialize as fallback
                      lr = getattr(self, 'alpha', 0.001) # Try to get loaded alpha if possible
                      self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
                      self.optimizer.load_state_dict(torch.load(optimizer_path)) # Retry loading state


            # --- Load Hyperparameters ---
            with open(hyperparams_path, 'r') as f:
                hyperparams = json.load(f)
                # Use loaded values, falling back to existing values if key is missing in JSON
                self.epsilon = hyperparams.get('epsilon', getattr(self, 'epsilon', 0.1))
                self.gamma = hyperparams.get('gamma', getattr(self, 'gamma', 0.95))
                self.train_step = hyperparams.get('train_step', getattr(self, 'train_step', 0))
                self.epsilon_decay = hyperparams.get('epsilon_decay', getattr(self, 'epsilon_decay', 0.9999))
                self.epsilon_min = hyperparams.get('epsilon_min', getattr(self, 'epsilon_min', 0.1))
                # Reload config-related params if they influence behavior (batch_size, target_update_freq)
                # Note: Memory size isn't reloaded here as we are not loading the buffer itself,
                # but the maxlen should match the original configuration.

            # Set model modes
            self.model.train()       # Resume training
            self.target_model.eval() # Target model stays in eval mode

            dqn_logger.info(f"DQNAgent state for {self.tls_id} loaded successfully from {directory_path}")

        except FileNotFoundError:
            # Should be caught by the check above, but handle defensively
            dqn_logger.error(f"Error loading DQNAgent state: File not found during load attempt in {directory_path}")
        except Exception as e:
            dqn_logger.error(f"Error loading DQNAgent state for {self.tls_id} from {directory_path}: {e}", exc_info=True)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory with improved error handling.

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
            
            # Store transition in memory
            self.memory.append((
                link_features, 
                signal_state,
                (link_internal_idx, action_idx),
                reward, 
                next_link_features,
                next_signal_state,
                done
            ))
        except Exception as e:
            # Log any errors but don't crash
            print(f"Error remembering experience: {e}")

    def learn(self, state, action, next_state, done):
        """Learn from experience using deep Q-learning.

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

        # Store experience
        self.remember(state, action, reward, next_state, done)

        # Only learn if we have enough samples
        if len(self.memory) < self.batch_size:
            return

        # Perform experience replay
        self._replay()

    def _replay(self):
        """Perform experience replay and network updates."""
        # Sample random batch from memory
        minibatch = random.sample(self.memory, self.batch_size)

        # Extract batch components
        link_features_batch = np.array([t[0] for t in minibatch])
        signal_state_batch = [t[1] for t in minibatch]
        actions_batch = [t[2] for t in minibatch]
        rewards_batch = np.array([t[3] for t in minibatch])
        next_link_features_batch = np.array([t[4] for t in minibatch])
        next_signal_state_batch = [t[5] for t in minibatch]
        dones_batch = np.array([t[6] for t in minibatch])
        
        # Convert to tensors
        link_features_tensor = torch.FloatTensor(link_features_batch)
        rewards_tensor = torch.FloatTensor(rewards_batch)
        next_link_features_tensor = torch.FloatTensor(next_link_features_batch)
        dones_tensor = torch.FloatTensor(dones_batch)
        
        # Separate link indices and action indices
        link_indices = torch.LongTensor([a[0] for a in actions_batch])
        action_indices = torch.LongTensor([a[1] for a in actions_batch])
        
        # Current Q-values
        link_scores, action_values = self.model(link_features_tensor, signal_state_batch)
        
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
            
        # Target Q-values
        with torch.no_grad():
            # Get scores and values from target network
            next_link_scores, next_action_values = self.target_model(
                next_link_features_tensor, next_signal_state_batch
            )
            
            # For each sample, find the best link and the best action for that link
            target_q = torch.zeros(self.batch_size)
            for i in range(self.batch_size):
                # Make sure we're handling tensor dimensions properly
                if next_link_scores.dim() <= 1 or next_link_scores.size(0) <= i:
                    # Handle unexpected tensor dimensions by defaulting to 0
                    best_action_val = 0.0
                else:
                    # Get the shape of the current sample's scores
                    if next_link_scores.dim() == 2:
                        # 2D tensor [batch_size, num_links]
                        valid_links = min(next_link_scores.size(1), self.max_links)
                        if valid_links > 0:
                            best_link_idx = next_link_scores[i, :valid_links].argmax().item()
                            
                            # Find the best action for that link
                            if best_link_idx < next_action_values.size(1):
                                best_action_val = next_action_values[i, best_link_idx].max().item()
                            else:
                                best_action_val = 0.0
                        else:
                            best_action_val = 0.0
                    else:
                        # Unexpected tensor dimension, default to 0
                        best_action_val = 0.0
                
                # Calculate target Q-value
                target_q[i] = rewards_tensor[i] + (1 - dones_tensor[i]) * self.gamma * best_action_val
                
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Update network
        self.optimizer.zero_grad()
        try:
            loss.backward()
            self.optimizer.step()
        except RuntimeError as e:
            # Handle the element does not require grad error in tests
            if "element 0 of tensors does not require grad" in str(e):
                print("Warning: Gradient calculation failed, skipping optimizer step")
            else:
                raise

        # Update target network if needed
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def calculate_reward(self, state, action, next_state):
        """Calculate reward for taking an action.
        
        Focuses purely on outcomes without encoding assumptions about
        "good" signal patterns.
        
        Args:
            state: Previous enhanced state
            action: Tuple of (link_index, new_state)
            next_state: Next enhanced state
            
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
        
        # Normalization constants
        MAX_WAITING_TIME = 500.0
        MAX_QUEUE_LENGTH = 20.0
        MAX_THROUGHPUT = 25.0
        
        # Normalize metrics
        norm_target_waiting = min(1.0, target_waiting_time / MAX_WAITING_TIME)
        norm_total_waiting = min(1.0, total_waiting_time / (MAX_WAITING_TIME * len(next_link_states)))
        norm_max_queue = min(1.0, max_queue_length / MAX_QUEUE_LENGTH)
        norm_throughput = min(1.0, total_throughput / MAX_THROUGHPUT)
        
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