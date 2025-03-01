import random
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from .agent import Agent


class AdvancedDQN(nn.Module):
    """Advanced DQN network with larger architecture for handling complex action spaces."""

    def __init__(self, state_size, action_size):
        """Initialize network architecture.

        Args:
            state_size: Size of the state input (number of features)
            action_size: Size of the action output (number of possible actions)
        """
        super(AdvancedDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 48)
        self.fc2 = nn.Linear(48, 48)
        self.fc3 = nn.Linear(48, action_size)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x: Input tensor

        Returns:
            Output tensor with Q-values for each action
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class AdvancedAgent(Agent):
    """
    Advanced agent using Deep Q-Learning with a combined action space
    that includes both phase selection and duration selection.
    """

    def __init__(self, possible_phases, duration_options, state_size,
                 alpha=0.001, gamma=0.95, epsilon=0.1, epsilon_decay=0.995,
                 epsilon_min=0.01, batch_size=32, memory_size=10000,
                 target_update_freq=100):
        """Initialize AdvancedAgent.

        Args:
            possible_phases: List of possible traffic light phases
            duration_options: List of possible duration values in seconds
            state_size: Size of the state vector
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_decay: Rate at which epsilon decreases
            epsilon_min: Minimum exploration rate
            batch_size: Batch size for learning
            memory_size: Size of experience replay memory
            target_update_freq: Frequency of target network updates
        """
        # Note: AdvancedAgent doesn't follow the standard Agent initialization
        # because it's designed to work with a combined action space
        self.tls_id = None  # Will be set when assigned to a traffic light
        self.network = None  # Will be set when assigned to a traffic light

        # Set device
        self.device = torch.device("cpu")  # Use CPU for consistency

        # Create combined action space: each action is a tuple (phase, duration)
        self.action_space = list(itertools.product(possible_phases, duration_options))
        self.action_size = len(self.action_space)
        self.state_size = state_size

        # DQN hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        # Experience replay memory
        self.memory = deque(maxlen=memory_size)

        # Initialize networks and move them to the device
        self.model = AdvancedDQN(state_size, self.action_size).to(self.device)
        self.target_model = AdvancedDQN(state_size, self.action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)

        # Target network update frequency
        self.target_update_freq = target_update_freq
        self.train_step = 0

        # Track last action for reward calculation
        self.last_action = None

    def set_traffic_light(self, tls_id, network):
        """Set the traffic light this agent controls.

        This method should be called after construction to associate
        the agent with a specific traffic light.

        Args:
            tls_id: ID of the traffic light
            network: Network object
        """
        self.tls_id = tls_id
        self.network = network

    def choose_action(self, state):
        """Choose an action based on the current state.

        Args:
            state: Current state observation vector

        Returns:
            Selected action tuple (phase, duration)
        """
        if np.random.rand() <= self.epsilon:
            return random.choice(self.action_space)
        else:
            # Convert state to tensor and move it to the device
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                q_values = self.model(state_tensor)

            action_idx = q_values.argmax().item()
            return self.action_space[action_idx]

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory.

        Args:
            state: Current state
            action: Action taken (tuple of phase, duration)
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Map the action tuple to its index in the action space
        try:
            action_idx = self.action_space.index(action)
        except ValueError:
            # If action not found (possible with NoAgent), use a default
            action_idx = 0

        # Store transition
        self.memory.append((state, action_idx, reward, next_state, done))

    def learn(self, state, action, next_state, done):
        """Learn from experience using deep Q-learning.

        Args:
            state: Previous state
            action: Action that was taken
            next_state: Resulting state
            done: Whether episode is done
        """
        # Calculate reward (if network is set)
        if self.network is not None:
            reward, _ = self.calculate_reward(state, action, next_state)
        else:
            # Default reward if network not set
            reward = 0

        # Store experience
        self.remember(state, action, reward, next_state, done)

        # Only learn if we have enough samples
        if len(self.memory) < self.batch_size:
            return

        # Perform experience replay
        self._replay()

    def _replay(self):
        """Perform experience replay and network updates."""
        # Sample random batch
        minibatch = random.sample(self.memory, self.batch_size)

        # Extract batch components and convert to tensors
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        actions = torch.LongTensor([t[1] for t in minibatch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.device)
        dones = torch.FloatTensor([t[4] for t in minibatch]).to(self.device)

        # Current Q-values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))

        # Target Q-values
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        target_q = target_q.unsqueeze(1)

        # Compute loss and update network
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update target network if needed
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def calculate_reward(self, state, action, next_state):
        """Calculate reward with advanced metrics.

        This implementation can be customized to reward
        efficient phase and duration choices.

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

        # Track current action for reward calculation
        self.last_action = action

        # Use parent's reward calculation but with additional weighting
        # for duration selection
        base_reward, components = super().calculate_reward(state, action, next_state)

        # Add duration efficiency component - reward shorter durations
        # when they're effective
        if isinstance(action, tuple) and len(action) > 1:
            phase, duration = action

            # Extract relevant metrics from state
            queue_length = state[2]
            vehicle_count = state[1]

            # Calculate duration efficiency
            # - Short durations are good for low traffic
            # - Longer durations are good for high traffic
            duration_efficiency = 0

            if vehicle_count > 0:
                optimal_duration = min(5 + (vehicle_count * 2), 20)  # Simple heuristic
                duration_diff = abs(duration - optimal_duration)
                # Penalize deviations from optimal duration
                duration_efficiency = -0.1 * (duration_diff / 20)

            # Add to components dictionary
            components['duration_efficiency'] = duration_efficiency

            # Add to total reward
            base_reward += duration_efficiency

        return base_reward, components