from abc import ABC, abstractmethod


class Agent(ABC):
    """Abstract base class for all agents.

    Each agent implementation should handle its own:
    1. State processing
    2. Action selection
    3. Learning mechanism
    4. Reward calculation
    """

    def __init__(self, tls_id, network):
        """Initialize the agent.

        Args:
            tls_id: ID of the traffic light this agent controls
            network: Network object providing access to simulation data
        """
        self.tls_id = tls_id
        self.network = network

    @abstractmethod
    def choose_action(self, state):
        """Choose an action based on the current state.

        Args:
            state: Current state observation

        Returns:
            The selected action
        """
        pass

    @abstractmethod
    def learn(self, state, action, next_state, done):
        """Learn from experience.

        Args:
            state: Previous state
            action: Action that was taken
            next_state: Resulting state
            done: Whether this is a terminal state
        """
        pass

    def calculate_reward(self, state, action, next_state):
        """Calculate the reward for taking an action.

        This default implementation can be overridden by subclasses
        to provide agent-specific reward functions.

        Args:
            state: Previous state
            action: Action that was taken
            next_state: Resulting state

        Returns:
            reward: Calculated reward value
            components: Dictionary of reward components (for analysis)
        """
        # Get traffic light data
        tls_id = self.tls_id
        lanes = self.network.get_controlled_lanes(tls_id)

        # Extract state features
        waiting_time = next_state[0]
        vehicle_count = next_state[1]
        queue_length = next_state[2]
        throughput = next_state[3]

        # Get per-lane metrics
        lane_queues = [self.network.get_lane_queue(lane) for lane in lanes]
        lane_waiting = [self.network.get_lane_waiting_time(lane) for lane in lanes]

        # Default normalization constants
        MAX_WAITING_TIME = 300.0
        MAX_THROUGHPUT = 20.0
        MAX_QUEUE_LENGTH = 15.0
        MAX_VEHICLES = 30.0

        # Default component weights
        W_WAITING = 0.40  # Waiting time (negative factor)
        W_THROUGHPUT = 0.30  # Throughput (positive factor)
        W_BALANCE = 0.15  # Lane balance (negative factor)
        W_CHANGE = 0.10  # Phase change (negative factor)
        W_CONGESTION = 0.05  # Overall congestion (negative factor)

        # Normalize metrics
        import numpy as np
        norm_waiting = min(1.0, waiting_time / MAX_WAITING_TIME)
        norm_throughput = min(1.0, throughput / MAX_THROUGHPUT)
        norm_queue = min(1.0, sum(lane_queues) / MAX_QUEUE_LENGTH)
        norm_vehicles = min(1.0, vehicle_count / MAX_VEHICLES)

        # Calculate queue imbalance
        if sum(lane_queues) > 0:
            cv_queue = np.std(lane_queues) / (np.mean(lane_queues) + 1e-6)
            norm_imbalance = min(1.0, cv_queue)
        else:
            norm_imbalance = 0.0

        # Phase change penalty - agents should track their last action
        last_action = getattr(self, 'last_action', None)
        phase_change = 1.0 if action != last_action else 0.0
        self.last_action = action

        # Calculate components
        waiting_penalty = -norm_waiting * W_WAITING
        throughput_reward = norm_throughput * W_THROUGHPUT
        balance_penalty = -norm_imbalance * W_BALANCE
        change_penalty = -phase_change * W_CHANGE
        congestion_penalty = -norm_vehicles * W_CONGESTION

        # Combine components
        total_reward = waiting_penalty + throughput_reward + balance_penalty + change_penalty + congestion_penalty

        # Return reward and components for analysis
        components = {
            'waiting_penalty': waiting_penalty,
            'throughput_reward': throughput_reward,
            'balance_penalty': balance_penalty,
            'change_penalty': change_penalty,
            'congestion_penalty': congestion_penalty
        }

        return total_reward, components