import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from src.agents.q_agent import QAgent


class TestQAgent(unittest.TestCase):
    """Test case for the QAgent class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock network for testing
        self.mock_network = MagicMock()
        self.mock_network.get_possible_phases.return_value = [0, 1, 2, 3]
        self.mock_network.get_controlled_lanes.return_value = ['lane1', 'lane2']
        self.mock_network.get_lane_queue.return_value = 5
        self.mock_network.get_lane_waiting_time.return_value = 30
        
        # Create an instance of QAgent
        self.agent = QAgent('tls1', self.mock_network, alpha=0.1, gamma=0.9, epsilon=0.2, state_bin_size=10)

    def test_initialization(self):
        """Test QAgent initialization."""
        self.assertEqual(self.agent.tls_id, 'tls1')
        self.assertEqual(self.agent.network, self.mock_network)
        self.assertEqual(self.agent.alpha, 0.1)
        self.assertEqual(self.agent.gamma, 0.9)
        self.assertEqual(self.agent.epsilon, 0.2)
        self.assertEqual(self.agent.state_bin_size, 10)
        self.assertEqual(self.agent.possible_actions, [0, 1, 2, 3])
        self.assertEqual(len(self.agent.q_values), 0)  # Q-values should start empty

    def test_discretize_state(self):
        """Test state discretization."""
        # Test with various input states
        state1 = np.array([25, 10, 5, 2])
        state2 = np.array([55, 15, 8, 4])
        
        # Discretization should divide by state_bin_size (10)
        self.assertEqual(self.agent._discretize_state(state1), 2)  # 25 // 10 = 2
        self.assertEqual(self.agent._discretize_state(state2), 5)  # 55 // 10 = 5

    @patch('random.random')
    @patch('random.choice')
    def test_choose_action_exploration(self, mock_choice, mock_random):
        """Test action selection during exploration."""
        # Setup mocks for exploration (epsilon = 0.2)
        mock_random.return_value = 0.1  # Below epsilon, should explore
        mock_choice.return_value = 2
        
        # Choose action
        action = self.agent.choose_action(np.array([30, 10, 5, 2]))
        
        # Verify exploration occurred
        mock_choice.assert_called_once_with([0, 1, 2, 3])
        self.assertEqual(action, 2)

    @patch('random.random')
    def test_choose_action_exploitation(self, mock_random):
        """Test action selection during exploitation."""
        # Setup mocks for exploitation (epsilon = 0.2)
        mock_random.return_value = 0.5  # Above epsilon, should exploit
        
        # Add some Q-values to the agent
        self.agent.q_values[(3, 0)] = 0.5
        self.agent.q_values[(3, 1)] = 0.2
        self.agent.q_values[(3, 2)] = 0.8  # Best action
        self.agent.q_values[(3, 3)] = 0.3
        
        # Choose action
        action = self.agent.choose_action(np.array([35, 10, 5, 2]))
        
        # Verify exploitation occurred (should choose action with highest Q-value)
        self.assertEqual(action, 2)

    def test_learn(self):
        """Test Q-learning update."""
        # Initial state and action
        state = np.array([30, 10, 5, 2])
        action = 2
        next_state = np.array([25, 8, 4, 3])
        done = False
        
        # Mock the reward calculation
        self.agent.calculate_reward = MagicMock(return_value=(0.5, {}))
        
        # Add initial Q-values
        discrete_state = self.agent._discretize_state(state)  # Should be 3
        discrete_next_state = self.agent._discretize_state(next_state)  # Should be 2
        
        self.agent.q_values[(discrete_state, action)] = 0.3
        self.agent.q_values[(discrete_next_state, 0)] = 0.4
        self.agent.q_values[(discrete_next_state, 1)] = 0.6  # Max Q for next state
        self.agent.q_values[(discrete_next_state, 2)] = 0.2
        self.agent.q_values[(discrete_next_state, 3)] = 0.1
        
        # Learn from this experience
        self.agent.learn(state, action, next_state, done)
        
        # Verify Q-value update
        # Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
        # Q(3,2) = 0.3 + 0.1 * (0.5 + 0.9 * 0.6 - 0.3) = 0.3 + 0.1 * (0.74) = 0.374
        expected_q = 0.3 + 0.1 * (0.5 + 0.9 * 0.6 - 0.3)
        self.assertAlmostEqual(self.agent.q_values[(discrete_state, action)], expected_q, places=6)

    def test_create_method(self):
        """Test the create class method."""
        # Call create with custom parameters
        result = QAgent.create('tls1', self.mock_network, alpha=0.2, epsilon=0.5)
        
        # Verify agent was created with correct parameters
        self.assertIsInstance(result, QAgent)
        self.assertEqual(result.tls_id, 'tls1')
        self.assertEqual(result.network, self.mock_network)
        self.assertEqual(result.alpha, 0.2)
        self.assertEqual(result.epsilon, 0.5)


if __name__ == '__main__':
    unittest.main()