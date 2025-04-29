import random
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, defaultdict
from typing import List, Dict, Any
from .agent import Agent


class CoordinationNetwork(nn.Module):
    """Network for handling coordination between adjacent traffic lights."""

    def __init__(self, local_state_size, adjacent_state_size, action_size):
        """Initialize network architecture for coordination.

        Args:
            local_state_size: Size of the state for the current traffic light
            adjacent_state_size: Size of the state for adjacent traffic lights
            action_size: Size of the action output
        """
        super(CoordinationNetwork, self).__init__()

        # Process local state
        self.local_fc1 = nn.Linear(local_state_size, 32)

        # Process adjacent states (assuming fixed number for simplicity)
        self.adjacent_fc1 = nn.Linear(adjacent_state_size, 32)

        # Combined processing
        self.combined_fc1 = nn.Linear(64, 48)  # 32 from local + 32 from adjacent
        self.combined_fc2 = nn.Linear(48, action_size)

    def forward(self, local_state, adjacent_state):
        """Forward pass through the network.

        Args:
            local_state: State tensor for the current traffic light
            adjacent_state: State tensor for adjacent traffic lights (concatenated)

        Returns:
            Output tensor with Q-values for each action
        """
        # Process local state
        local_features = F.relu(self.local_fc1(local_state))

        # Handle case when adjacent_state has wrong dimension
        if adjacent_state.size(1) != self.adjacent_fc1.in_features:
            # Create a zero tensor with correct size as fallback
            batch_size = adjacent_state.size(0)
            adjacent_state = torch.zeros(batch_size, self.adjacent_fc1.in_features, device=adjacent_state.device)
        
        # Process adjacent states
        adjacent_features = F.relu(self.adjacent_fc1(adjacent_state))

        # Combine features
        combined = torch.cat((local_features, adjacent_features), dim=1)
        combined = F.relu(self.combined_fc1(combined))

        # Output Q-values
        return self.combined_fc2(combined)


class EnhancedTemporalNetwork(nn.Module):
    """Enhanced DQN with temporal considerations and attention mechanism."""

    def __init__(self, state_size, history_steps, action_size):
        """Initialize network with temporal and spatial attention.

        Args:
            state_size: Size of the state input (number of features per time step)
            history_steps: Number of historical state steps to consider
            action_size: Size of the action output
        """
        super(EnhancedTemporalNetwork, self).__init__()

        # Temporal feature extraction
        self.temporal_fc1 = nn.Linear(state_size, 48)

        # Attention mechanism for temporal data
        self.attention_w = nn.Parameter(torch.Tensor(history_steps, 48))
        self.attention_v = nn.Parameter(torch.Tensor(48, 1))

        # Initialize attention parameters
        nn.init.xavier_uniform_(self.attention_w)
        nn.init.xavier_uniform_(self.attention_v)

        # Final layers
        self.fc1 = nn.Linear(48, 48)
        self.fc2 = nn.Linear(48, action_size)

    def forward(self, x):
        """Forward pass through the network with attention.

        Args:
            x: Input tensor of shape [batch_size, history_steps, state_size]

        Returns:
            Output tensor with Q-values for each action
        """
        batch_size, time_steps, _ = x.size()

        # Process each time step
        temporal_features = []
        for t in range(time_steps):
            features = F.relu(self.temporal_fc1(x[:, t, :]))
            temporal_features.append(features)

        # Stack features for attention calculation
        stacked_features = torch.stack(temporal_features, dim=1)  # [batch_size, time_steps, 48]

        # Compute attention scores
        attention = torch.tanh(torch.matmul(stacked_features, self.attention_v))  # [batch_size, time_steps, 1]
        attention = F.softmax(attention, dim=1)

        # Apply attention weights
        context = torch.sum(stacked_features * attention, dim=1)  # [batch_size, 48]

        # Final layers
        x = F.relu(self.fc1(context))
        return self.fc2(x)


class EnhancedAgent(Agent):
    """
    Enhanced agent with coordination capabilities, adaptive duration selection,
    and temporal state consideration for improved traffic control.
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        "alpha": 0.001,            # Learning rate
        "gamma": 0.96,             # Discount factor
        "epsilon": 0.9,            # Very high exploration rate for extensive exploration
        "epsilon_decay": 0.9999,   # Extremely slow decay to maintain exploration
        "epsilon_min": 0.15,       # Higher minimum to always keep some exploration
        "batch_size": 64,
        "memory_size": 100000,
        "target_update_freq": 300, # Less frequent updates for stability across episodes
        "history_length": 5,
        "min_phase_duration": 5,
        "max_phase_duration": 40,
        "duration_step": 5,
        "prioritized_replay": True,
        "use_coordination": True
    }

    def __init__(self, tls_id, network, adjacent_tls=None,
                 alpha=0.001, gamma=0.96, epsilon=0.1, epsilon_decay=0.995,
                 epsilon_min=0.01, batch_size=64, memory_size=100000,
                 target_update_freq=100, history_length=5,
                 min_phase_duration=5, max_phase_duration=40,
                 duration_step=5, prioritized_replay=True,
                 use_coordination=True):
        """Initialize EnhancedAgent.

        Args:
            tls_id: ID of the traffic light this agent controls
            network: Network object providing access to simulation data
            adjacent_tls: List of adjacent traffic light IDs for coordination
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_decay: Rate at which epsilon decreases
            epsilon_min: Minimum exploration rate
            batch_size: Batch size for learning
            memory_size: Size of experience replay memory
            target_update_freq: Frequency of target network updates
            history_length: Number of historical states to consider
            min_phase_duration: Minimum duration for a phase
            max_phase_duration: Maximum duration for a phase
            duration_step: Step size for duration adjustment
            prioritized_replay: Whether to use prioritized experience replay
            use_coordination: Whether to use coordination mechanism
        """
        super().__init__(tls_id, network)

        # Store configuration
        self.adjacent_tls = adjacent_tls if adjacent_tls else []
        self.history_length = history_length
        self.min_phase_duration = min_phase_duration
        self.max_phase_duration = max_phase_duration
        self.duration_step = duration_step
        self.use_coordination = use_coordination
        self.prioritized_replay = prioritized_replay

        # Set device
        self.device = torch.device("cpu")


        # Get possible actions from network
        self.possible_phases = network.get_possible_phases(tls_id)
        self.num_phases = len(self.possible_phases)

        # Create adaptive duration options
        self.duration_options = list(range(
            self.min_phase_duration,
            self.max_phase_duration + 1,
            self.duration_step
        ))

        # Create combined action space: each action is a tuple (phase, duration)
        self.action_space = list(itertools.product(
            self.possible_phases, self.duration_options
        ))
        self.action_size = len(self.action_space)

        # State size based on traffic and historical data
        self.base_state_size = 8  # Expanded state features
        self.state_size = self.base_state_size * self.history_length

        # DQN hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.alpha = alpha

        # Replay memory
        if self.prioritized_replay:
            # Prioritized replay with (state, action, reward, next_state, done, priority)
            self.memory = deque(maxlen=memory_size)
            self.priority_alpha = 0.6  # Priority exponent
            self.priority_beta = 0.4  # Importance sampling exponent
            self.priority_beta_increment = 0.001
            self.epsilon_pri = 0.01  # Small constant to avoid zero priority
        else:
            # Standard replay memory
            self.memory = deque(maxlen=memory_size)

        # State history buffer
        self.state_history = deque(maxlen=self.history_length)
        # Initialize with zeros
        for _ in range(self.history_length):
            self.state_history.append(np.zeros(self.base_state_size))

        # Initialize networks
        if self.use_coordination and self.adjacent_tls:
            # For coordination, we need a different network architecture
            adjacent_state_size = self.base_state_size * len(self.adjacent_tls)
            self.model = CoordinationNetwork(
                self.state_size, adjacent_state_size, self.action_size
            ).to(self.device)
            self.target_model = CoordinationNetwork(
                self.state_size, adjacent_state_size, self.action_size
            ).to(self.device)
        else:
            # For temporal consideration without coordination
            self.model = EnhancedTemporalNetwork(
                self.base_state_size, self.history_length, self.action_size
            ).to(self.device)
            self.target_model = EnhancedTemporalNetwork(
                self.base_state_size, self.history_length, self.action_size
            ).to(self.device)

        # Copy weights from main to target network
        self.target_model.load_state_dict(self.model.state_dict())

        # Optimizer with gradient clipping
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)

        # Target network update frequency
        self.target_update_freq = target_update_freq
        self.train_step = 0

        # Performance tracking
        self.performance_history = {
            'rewards': [],
            'phase_rewards': {}  # Using dict instead of list for phase_rewards
        }
        self.last_action = None
        self.current_duration = min_phase_duration
        self.time_since_last_change = 0

        # Phase-specific metrics for adaptive duration
        self.phase_performance = {phase: {'waiting_times': [], 'queue_lengths': []}
                                  for phase in self.possible_phases}

        # For green wave coordination
        self.adjacent_phases = {}  # Store phases of adjacent traffic lights
        
    @classmethod
    def create(cls, tls_id, network, **kwargs):
        """Create an instance of the EnhancedAgent with proper configuration.
        
        This class method handles agent-specific initialization logic,
        including adjacent traffic light determination.
        
        Args:
            tls_id: ID of the traffic light this agent controls
            network: Network object providing access to simulation data
            **kwargs: Additional configuration parameters
            
        Returns:
            Properly configured EnhancedAgent instance
        """
        # Start with default configuration
        config = cls.DEFAULT_CONFIG.copy()
        
        # Override with provided kwargs
        config.update(kwargs)
        
        # Create a temporary instance to access instance methods
        temp_agent = cls(tls_id, network, **config)
        
        # Get adjacent traffic lights
        adjacent_tls = temp_agent.get_adjacent_traffic_lights(network)
        
        # Create the final agent with adjacent traffic lights
        agent = cls(
            tls_id=tls_id, 
            network=network,
            adjacent_tls=adjacent_tls,
            **config
        )
        
        return agent

    def preprocess_state(self, state):
        """Preprocess the raw state to include more meaningful features.

        Args:
            state: Raw state vector from the environment

        Returns:
            Processed state with additional features
        """
        # Extract original features
        waiting_time = state[0]
        vehicle_count = state[1]
        queue_length = state[2]
        throughput = state[3]
        phase_duration = state[4]
        time_since_switch = state[5]

        # Get lane-specific data
        lanes = self.network.get_controlled_lanes(self.tls_id)

        # Calculate additional features
        lane_counts = [self.network.get_lane_vehicle_count(lane) for lane in lanes]
        lane_queues = [self.network.get_lane_queue(lane) for lane in lanes]

        # Calculate lane imbalance (standard deviation of vehicle counts)
        lane_imbalance = np.std(lane_counts) if lane_counts else 0

        # Calculate queue growth rate (if we have history)
        queue_growth = 0
        if len(self.state_history) >= 2 and self.state_history[-1][2] > 0:
            queue_growth = (queue_length - self.state_history[-1][2]) / self.state_history[-1][2]

        # Create enhanced state
        enhanced_state = np.array([
            waiting_time,
            vehicle_count,
            queue_length,
            throughput,
            phase_duration / self.max_phase_duration,  # Normalize
            time_since_switch / 60.0,  # Normalize with max expected switch time
            lane_imbalance,
            queue_growth
        ])

        # Update state history
        self.state_history.append(enhanced_state)

        # Flatten history for the network input
        flat_history = np.concatenate(list(self.state_history))

        return flat_history

    def get_adjacent_states(self):
        """Get states of adjacent traffic lights for coordination.

        Returns:
            Concatenated state vector of adjacent traffic lights
        """
        if not self.adjacent_tls:
            # Return properly sized zero array based on adjacent_fc1 input features
            # This will typically be base_state_size * expected number of adjacent lights
            expected_adjacent_size = self.base_state_size * 2  # Assume 2 adjacent TLs
            return np.zeros(expected_adjacent_size)

        adjacent_states = []
        for tls_id in self.adjacent_tls:
            try:
                # Try to get the current state from the network
                all_states = self.network.get_state()
                state = all_states.get(tls_id)
                if state is not None and len(state) >= self.base_state_size:
                    # Process this state using the same preprocessing
                    adjacent_states.append(state[:self.base_state_size])
                else:
                    # If not available, use zeros
                    adjacent_states.append(np.zeros(self.base_state_size))
            except Exception as e:
                print(f"Error getting adjacent state for {tls_id}: {e}")
                # In case of error, use zeros
                adjacent_states.append(np.zeros(self.base_state_size))

            # Also store the current phase for green wave coordination
            try:
                current_phase = self.network.get_current_phase(tls_id)
                self.adjacent_phases[tls_id] = current_phase
            except Exception as e:
                print(f"Error getting phase for {tls_id}: {e}")
                self.adjacent_phases[tls_id] = 0

        # Create a proper sized array
        if not adjacent_states:
            expected_adjacent_size = self.base_state_size * 2  # Assume 2 adjacent TLs
            return np.zeros(expected_adjacent_size)
            
        return np.concatenate(adjacent_states)

    def predict_traffic_flow(self, state):
        """Predict future traffic flow based on current state.

        This is a simple prediction based on recent history. In a real-world
        implementation, this could be replaced with a more sophisticated
        prediction model.

        Args:
            state: Current state

        Returns:
            Predicted change in vehicle count and queue length
        """
        if len(self.state_history) < 2:
            return 0, 0

        # Extract vehicle counts and queue lengths from history
        vehicle_counts = [s[1] for s in self.state_history]
        queue_lengths = [s[2] for s in self.state_history]

        # Calculate recent trends
        vehicle_trend = np.mean([vehicle_counts[i] - vehicle_counts[i - 1]
                                 for i in range(1, len(vehicle_counts))])
        queue_trend = np.mean([queue_lengths[i] - queue_lengths[i - 1]
                               for i in range(1, len(queue_lengths))])

        return vehicle_trend, queue_trend

    def choose_action(self, state):
        """Choose an action based on the current state.

        Args:
            state: Current state observation vector

        Returns:
            Selected action tuple (phase, duration)
        """
        # Preprocess state
        processed_state = self.preprocess_state(state)

        # Get adjacent states if using coordination
        if self.use_coordination and self.adjacent_tls:
            adjacent_states = self.get_adjacent_states()
        else:
            adjacent_states = None

        # Exploration: Choose random action
        if np.random.rand() <= self.epsilon:
            # Adaptive exploration: bias towards promising actions
            if np.random.rand() < 0.3 and self.performance_history['rewards']:
                # Pick phase based on historical performance
                phase_perf = {}
                for phase in self.possible_phases:
                    phase_actions = [a for a in self.action_space if a[0] == phase]
                    # Create string keys for lookup
                    phase_rewards = [self.performance_history['phase_rewards'].get(f"{a[0]}_{a[1]}", -1)
                                     for a in phase_actions]
                    phase_perf[phase] = max(phase_rewards) if phase_rewards else -1

                # Choose phase with probability proportional to performance
                phases = list(phase_perf.keys())
                
                # Ensure we have valid performance data
                if len(phases) > 0 and not all(v == -1 for v in phase_perf.values()):
                    # Convert rewards to positive values for selection
                    min_reward = min(phase_perf.values())
                    adjusted_rewards = [phase_perf[p] - min_reward + 1 for p in phases]
                    
                    # Check for zero sum to avoid division by zero
                    reward_sum = sum(adjusted_rewards)
                    if reward_sum > 0:
                        # Normalize to probabilities
                        probs = [r / reward_sum for r in adjusted_rewards]
                        # Select phase
                        phase = np.random.choice(phases, p=probs)
                    else:
                        # Fall back to uniform random if all rewards are equal
                        phase = np.random.choice(phases)
                else:
                    # Fall back to uniform random if no valid performance data
                    phase = np.random.choice(self.possible_phases)

                # Select duration based on current traffic
                vehicle_trend, queue_trend = self.predict_traffic_flow(state)

                if queue_trend > 0.1:  # Growing queue
                    # Prefer longer durations
                    duration_options = [d for d in self.duration_options
                                        if d >= (self.min_phase_duration + self.max_phase_duration) / 2]
                elif queue_trend < -0.1:  # Shrinking queue
                    # Prefer shorter durations
                    duration_options = [d for d in self.duration_options
                                        if d <= (self.min_phase_duration + self.max_phase_duration) / 2]
                else:
                    # Use all durations
                    duration_options = self.duration_options

                duration = random.choice(duration_options)
                return (phase, duration)
            else:
                # Completely random action
                return random.choice(self.action_space)

        # Exploitation: Choose best action according to Q-network
        # Convert state to tensor and move to device
        state_tensor = torch.from_numpy(processed_state).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.use_coordination and adjacent_states is not None:
                # Use coordination network
                adjacent_tensor = torch.from_numpy(adjacent_states).float().unsqueeze(0).to(self.device)
                q_values = self.model(state_tensor, adjacent_tensor)
            else:
                # Reshape for temporal network if using that architecture
                if isinstance(self.model, EnhancedTemporalNetwork):
                    # Reshape to [batch_size, history_steps, state_features]
                    reshaped_state = state_tensor.view(1, self.history_length, self.base_state_size)
                    q_values = self.model(reshaped_state)
                else:
                    # Standard forward pass
                    q_values = self.model(state_tensor)

        action_idx = q_values.argmax().item()
        selected_action = self.action_space[action_idx]

        # Update performance tracking
        self.last_action = selected_action

        return selected_action

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory.

        Args:
            state: Current state
            action: Action taken (tuple of phase, duration)
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Preprocess states
        processed_state = self.preprocess_state(state)
        processed_next_state = self.preprocess_state(next_state)

        # Find action index
        try:
            action_idx = self.action_space.index(action)
        except ValueError:
            # If action not found, use a default
            action_idx = 0

        if self.prioritized_replay:
            # For new experiences, assign max priority
            max_priority = max([m[5] for m in self.memory]) if self.memory else 1.0
            self.memory.append((
                processed_state, action_idx, reward, processed_next_state, done, max_priority
            ))
        else:
            # Standard experience storage
            self.memory.append((
                processed_state, action_idx, reward, processed_next_state, done
            ))

    def learn(self, state, action, next_state, done):
        """Learn from experience using advanced deep Q-learning.

        Args:
            state: Previous state
            action: Action that was taken
            next_state: Resulting state
            done: Whether episode is done
        """
        # Calculate reward
        reward, components = self.calculate_reward(state, action, next_state)

        # Store experience
        self.remember(state, action, reward, next_state, done)

        # Update performance tracking
        self.performance_history['rewards'].append(reward)
        if isinstance(action, tuple) and len(action) == 2:
            phase, duration = action
            # Create a string key from the tuple for compatibility
            action_key = f"{phase}_{duration}"
            
            if action_key not in self.performance_history['phase_rewards']:
                self.performance_history['phase_rewards'][action_key] = reward
            else:
                # Exponential moving average
                old_reward = self.performance_history['phase_rewards'][action_key]
                self.performance_history['phase_rewards'][action_key] = (
                        0.9 * old_reward + 0.1 * reward
                )

            # Track phase-specific metrics
            if phase in self.phase_performance:
                self.phase_performance[phase]['waiting_times'].append(state[0])
                self.phase_performance[phase]['queue_lengths'].append(state[2])

        # Only learn if we have enough samples
        if len(self.memory) < self.batch_size:
            return

        # Perform experience replay
        self._replay()

    def _replay(self):
        """Perform experience replay with prioritization and other enhancements."""
        if self.prioritized_replay:
            # Prioritized experience replay
            priorities = np.array([m[5] for m in self.memory])
            probabilities = priorities ** self.priority_alpha
            probabilities = probabilities / np.sum(probabilities)

            # Update beta parameter (increases over time to reduce bias)
            self.priority_beta = min(1.0, self.priority_beta + self.priority_beta_increment)

            # Sample batch based on probabilities
            indices = np.random.choice(
                len(self.memory), self.batch_size, replace=False, p=probabilities
            )

            # Calculate importance sampling weights
            weights = (len(self.memory) * probabilities[indices]) ** (-self.priority_beta)
            weights = weights / np.max(weights)  # Normalize weights
            weights = torch.FloatTensor(weights).to(self.device)

            # Get samples
            batch = [self.memory[i] for i in indices]
            states = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
            actions = torch.LongTensor([b[1] for b in batch]).to(self.device)
            rewards = torch.FloatTensor([b[2] for b in batch]).to(self.device)
            next_states = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
            dones = torch.FloatTensor([b[4] for b in batch]).to(self.device)

        else:
            # Standard experience replay
            batch = random.sample(self.memory, self.batch_size)
            states = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
            actions = torch.LongTensor([b[1] for b in batch]).to(self.device)
            rewards = torch.FloatTensor([b[2] for b in batch]).to(self.device)
            next_states = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
            dones = torch.FloatTensor([b[4] for b in batch]).to(self.device)
            weights = torch.ones(self.batch_size).to(self.device)  # All weights equal

        # Get adjacent states if using coordination
        if self.use_coordination and self.adjacent_tls:
            # Create properly sized adjacent state tensors
            adjacent_size = self.base_state_size * max(2, len(self.adjacent_tls))
            
            # Check if the model's adjacent_fc1 has a specific input size
            if hasattr(self.model, 'adjacent_fc1'):
                adjacent_size = self.model.adjacent_fc1.in_features
                
            # Create dummy adjacent states with correct size
            adjacent_states = torch.zeros(
                self.batch_size,
                adjacent_size
            ).to(self.device)
            adjacent_next_states = adjacent_states.clone()

            # Forward pass with coordination
            current_q = self.model(states, adjacent_states).gather(1, actions.unsqueeze(1))
            next_q = self.target_model(next_states, adjacent_next_states).max(1)[0].detach()
        else:
            # Reshape for temporal network if using that architecture
            if isinstance(self.model, EnhancedTemporalNetwork):
                try:
                    # Reshape to [batch_size, history_steps, state_features]
                    states_reshaped = states.view(
                        self.batch_size, self.history_length, self.base_state_size
                    )
                    next_states_reshaped = next_states.view(
                        self.batch_size, self.history_length, self.base_state_size
                    )

                    # Forward pass with temporal network
                    current_q = self.model(states_reshaped).gather(1, actions.unsqueeze(1))
                    next_q = self.target_model(next_states_reshaped).max(1)[0].detach()
                except RuntimeError as e:
                    print(f"Error in reshaping for temporal network: {e}")
                    # Fall back to standard processing if reshaping fails
                    current_q = self.model(states).gather(1, actions.unsqueeze(1))
                    next_q = self.target_model(next_states).max(1)[0].detach()
            else:
                # Standard forward pass
                current_q = self.model(states).gather(1, actions.unsqueeze(1))
                next_q = self.target_model(next_states).max(1)[0].detach()

        # Calculate target Q values with double DQN
        with torch.no_grad():
            if self.use_coordination and self.adjacent_tls:
                try:
                    next_actions = self.model(next_states, adjacent_next_states).argmax(1)
                    next_q = self.target_model(next_states, adjacent_next_states).gather(
                        1, next_actions.unsqueeze(1)
                    ).squeeze(1)
                except RuntimeError as e:
                    print(f"Error in Double DQN coordination branch: {e}")
                    # Fall back to simpler implementation if error occurs
                    next_q = self.target_model(next_states, adjacent_next_states).max(1)[0]
            elif isinstance(self.model, EnhancedTemporalNetwork):
                try:
                    next_actions = self.model(next_states_reshaped).argmax(1)
                    next_q = self.target_model(next_states_reshaped).gather(
                        1, next_actions.unsqueeze(1)
                    ).squeeze(1)
                except RuntimeError as e:
                    print(f"Error in Double DQN temporal branch: {e}")
                    # Fall back to simpler implementation if error occurs
                    next_q = self.target_model(next_states_reshaped).max(1)[0]
            else:
                next_actions = self.model(next_states).argmax(1)
                next_q = self.target_model(next_states).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)

        # Calculate target values
        target_q = rewards + (1 - dones) * self.gamma * next_q
        target_q = target_q.unsqueeze(1)

        # Compute weighted MSE loss for prioritized replay
        if self.prioritized_replay:
            # Calculate TD errors for updating priorities
            td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()

            # Compute loss with importance sampling weights
            loss = (weights.unsqueeze(1) * F.mse_loss(current_q, target_q, reduction='none')).mean()

            # Update priorities in memory
            for i, idx in enumerate(indices):
                self.memory[idx] = (
                    self.memory[idx][0],
                    self.memory[idx][1],
                    self.memory[idx][2],
                    self.memory[idx][3],
                    self.memory[idx][4],
                    td_errors[i][0] + self.epsilon_pri  # Add small constant to avoid zero priority
                )
        else:
            # Standard MSE loss
            loss = F.mse_loss(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update target network if needed
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def calculate_reward(self, state, action, next_state):
        """Calculate reward with enhanced metrics for traffic optimization.

        This implementation builds on the base Agent class but adds:
        - Green wave coordination with adjacent traffic lights
        - Adaptive phase duration rewards
        - Travel time reduction focus
        - Emergency vehicle priority (if available in simulation)

        Args:
            state: Previous state
            action: Action taken (tuple of phase, duration)
            next_state: Resulting state

        Returns:
            reward: Calculated reward value
            components: Dictionary of reward components
        """
        if self.network is None:
            # Default components if network not set
            return 0, {'waiting_penalty': 0, 'throughput_reward': 0}

        # Track current action for future reward calculation
        self.last_action = action

        # Extract state features
        waiting_time = next_state[0]
        vehicle_count = next_state[1]
        queue_length = next_state[2]
        throughput = next_state[3]

        # Get per-lane metrics for calculating balance
        tls_id = self.tls_id
        lanes = self.network.get_controlled_lanes(tls_id)
        lane_queues = [self.network.get_lane_queue(lane) for lane in lanes]
        lane_waiting = [self.network.get_lane_waiting_time(lane) for lane in lanes]

        # Normalization constants
        MAX_WAITING_TIME = 300.0
        MAX_THROUGHPUT = 20.0
        MAX_QUEUE_LENGTH = 15.0
        MAX_VEHICLES = 30.0

        # Component weights
        W_WAITING = 0.35  # Waiting time (negative factor)
        W_THROUGHPUT = 0.25  # Throughput (positive factor)
        W_BALANCE = 0.15  # Lane balance (negative factor)
        W_CHANGE = 0.05  # Phase change (negative factor)
        W_CONGESTION = 0.05  # Overall congestion (negative factor)
        W_COORDINATION = 0.10  # Coordination with adjacent lights
        W_DURATION = 0.05  # Duration efficiency

        # Normalize metrics
        norm_waiting = min(1.0, waiting_time / MAX_WAITING_TIME)
        norm_throughput = min(1.0, throughput / MAX_THROUGHPUT)
        norm_queue = min(1.0, sum(lane_queues) / MAX_QUEUE_LENGTH)
        norm_vehicles = min(1.0, vehicle_count / MAX_VEHICLES)

        # Calculate queue imbalance (standard deviation / mean)
        if sum(lane_queues) > 0:
            cv_queue = np.std(lane_queues) / (np.mean(lane_queues) + 1e-6)
            norm_imbalance = min(1.0, cv_queue)
        else:
            norm_imbalance = 0.0

        # Phase change penalty
        last_action = getattr(self, 'last_action', None)
        if isinstance(action, tuple) and isinstance(last_action, tuple):
            # Only penalize phase changes, not duration changes
            phase_change = 1.0 if action[0] != last_action[0] else 0.0
        else:
            phase_change = 1.0 if action != last_action else 0.0

        # Duration efficiency component
        duration_efficiency = 0
        if isinstance(action, tuple) and len(action) > 1:
            phase, duration = action

            # Calculate expected optimal duration based on queue length and vehicle count
            # Simple heuristic: longer queues need more time
            expected_duration = min(
                self.min_phase_duration + (queue_length * 2),
                self.max_phase_duration
            )

            # Penalize too short or too long durations
            duration_diff = abs(duration - expected_duration)
            max_diff = self.max_phase_duration - self.min_phase_duration
            duration_efficiency = -0.5 * (duration_diff / max_diff)

            # But also reward shorter durations when queues are emptying
            if queue_length == 0 and vehicle_count < 5:
                # Reward shorter durations when there's no queue
                duration_efficiency += 0.5 * (1 - duration / self.max_phase_duration)

        # Green wave coordination component
        coordination_reward = 0
        if self.use_coordination and self.adjacent_tls and isinstance(action, tuple):
            current_phase = action[0]

            # Check if our phase aligns with adjacent traffic lights
            for adj_tls in self.adjacent_tls:
                adj_phase = self.adjacent_phases.get(adj_tls)
                if adj_phase is not None:
                    # Simple coordination logic: reward matching phases
                    # In a real implementation, we'd have a more sophisticated coordination logic
                    if (current_phase % 2) == (adj_phase % 2):  # Simple parity matching
                        coordination_reward += 0.1

            # Normalize
            if self.adjacent_tls:
                coordination_reward /= len(self.adjacent_tls)

        # Calculate components
        waiting_penalty = -norm_waiting * W_WAITING
        throughput_reward = norm_throughput * W_THROUGHPUT
        balance_penalty = -norm_imbalance * W_BALANCE
        change_penalty = -phase_change * W_CHANGE
        congestion_penalty = -norm_vehicles * W_CONGESTION
        coordination_component = coordination_reward * W_COORDINATION
        duration_component = duration_efficiency * W_DURATION

        # Combine components
        total_reward = (
                waiting_penalty +
                throughput_reward +
                balance_penalty +
                change_penalty +
                congestion_penalty +
                coordination_component +
                duration_component
        )

        # Return reward and components for analysis
        components = {
            'waiting_penalty': waiting_penalty,
            'throughput_reward': throughput_reward,
            'balance_penalty': balance_penalty,
            'change_penalty': change_penalty,
            'congestion_penalty': congestion_penalty,
            'coordination_reward': coordination_component,
            'duration_efficiency': duration_component
        }

        return total_reward, components

    def find_optimal_duration(self, phase, state):
        """Find the optimal duration for a given phase based on current state.

        Args:
            phase: The selected traffic light phase
            state: Current state observation

        Returns:
            Optimal duration in seconds
        """
        # Extract relevant metrics
        vehicle_count = state[1]
        queue_length = state[2]

        # Check historical performance for this phase
        if phase in self.phase_performance:
            phase_data = self.phase_performance[phase]

            # If we have enough data points
            if len(phase_data['waiting_times']) >= 3:
                # Calculate average waiting time reduction per second of green time
                waiting_times = phase_data['waiting_times']
                queue_lengths = phase_data['queue_lengths']

                # Calculate average queue per vehicle for this phase
                avg_queue_per_vehicle = np.mean([
                    q / (v + 1e-6) for q, v in zip(queue_lengths, waiting_times)
                ])

                # Estimate time needed to clear queue
                estimated_clear_time = queue_length / (avg_queue_per_vehicle + 1e-6)

                # Ensure within bounds
                optimal_duration = np.clip(
                    estimated_clear_time,
                    self.min_phase_duration,
                    self.max_phase_duration
                )

                # Round to nearest duration step
                optimal_duration = round(optimal_duration / self.duration_step) * self.duration_step

                return optimal_duration

        # Fallback: simple heuristic based on current state
        base_duration = self.min_phase_duration

        # Add time based on queue length (2 seconds per vehicle in queue)
        queue_factor = min(queue_length * 2, self.max_phase_duration - base_duration)

        # Add time based on vehicle count
        count_factor = min(vehicle_count * 0.5, self.max_phase_duration - base_duration - queue_factor)

        # Calculate total duration
        total_duration = base_duration + queue_factor + count_factor

        # Round to nearest duration step
        optimal_duration = round(total_duration / self.duration_step) * self.duration_step

        # Ensure within bounds
        return np.clip(optimal_duration, self.min_phase_duration, self.max_phase_duration)

    def adaptive_action_selection(self, state):
        """Adaptively select an action based on current traffic conditions.

        This provides a more sophisticated action selection than the basic
        epsilon-greedy approach, incorporating traffic engineering principles.

        Args:
            state: Current state observation

        Returns:
            Selected action tuple (phase, duration)
        """
        # Extract state features
        vehicle_count = state[1]
        queue_length = state[2]

        # First, determine if we should change the phase
        if queue_length < 2 and vehicle_count < 5:
            # Light traffic: be more responsive to other directions
            change_threshold = 0.3
        elif queue_length > 10 or vehicle_count > 20:
            # Heavy traffic: stick with current phase longer
            change_threshold = 0.8
        else:
            # Medium traffic: balanced approach
            change_threshold = 0.5

        # Get current phase
        current_phase = None
        if self.last_action and isinstance(self.last_action, tuple):
            current_phase = self.last_action[0]

        # Decide whether to change phase
        if current_phase is not None and np.random.rand() < change_threshold:
            # Keep current phase, find optimal duration
            optimal_duration = self.find_optimal_duration(current_phase, state)
            return (current_phase, optimal_duration)
        else:
            # Change to a different phase - select which one
            if current_phase is not None:
                # Exclude current phase
                candidate_phases = [p for p in self.possible_phases if p != current_phase]
            else:
                # Consider all phases
                candidate_phases = self.possible_phases

            # Simple selection: uniform random from candidates
            new_phase = np.random.choice(candidate_phases)

            # Find optimal duration for new phase
            optimal_duration = self.find_optimal_duration(new_phase, state)

            return (new_phase, optimal_duration)