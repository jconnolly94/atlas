import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from src.agents.dqn_agent import DQNAgent, DQN


class TestDQNAgent(unittest.TestCase):
    """Test case for the DQNAgent class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock network for testing
        self.mock_network = MagicMock()
        self.mock_network.get_possible_phases.return_value = [0, 1, 2, 3]
        self.mock_network.get_controlled_lanes.return_value = ['lane1', 'lane2']
        self.mock_network.get_lane_queue.return_value = 5
        self.mock_network.get_lane_waiting_time.return_value = 30
        
        # Create agent with test parameters
        self.agent = DQNAgent(
            'tls1', 
            self.mock_network, 
            alpha=0.01, 
            gamma=0.9, 
            epsilon=0.3,
            epsilon_decay=0.99,
            epsilon_min=0.05,
            batch_size=16,
            memory_size=1000,
            target_update_freq=50
        )

    def test_initialization(self):
        """Test DQNAgent initialization."""
        self.assertEqual(self.agent.tls_id, 'tls1')
        self.assertEqual(self.agent.network, self.mock_network)
        self.assertEqual(self.agent.gamma, 0.9)
        self.assertEqual(self.agent.epsilon, 0.3)
        self.assertEqual(self.agent.epsilon_decay, 0.99)
        self.assertEqual(self.agent.epsilon_min, 0.05)
        self.assertEqual(self.agent.batch_size, 16)
        self.assertEqual(self.agent.possible_actions, [0, 1, 2, 3])
        self.assertEqual(self.agent.action_size, 4)
        self.assertEqual(self.agent.state_size, 6)
        self.assertEqual(len(self.agent.memory), 0)
        self.assertEqual(self.agent.train_step, 0)
        
        # Verify network creation
        self.assertIsInstance(self.agent.model, DQN)
        self.assertIsInstance(self.agent.target_model, DQN)

    @patch('numpy.random.rand')
    @patch('random.choice')
    def test_choose_action_exploration(self, mock_choice, mock_rand):
        """Test action selection during exploration."""
        # Setup for exploration
        mock_rand.return_value = 0.1  # Below epsilon=0.3, should explore
        mock_choice.return_value = 2
        
        # Choose action
        action = self.agent.choose_action(np.array([30, 10, 5, 2, 1, 0]))
        
        # Verify exploration
        mock_choice.assert_called_once_with([0, 1, 2, 3])
        self.assertEqual(action, 2)

    @patch('numpy.random.rand')
    def test_choose_action_exploitation(self, mock_rand):
        """Test action selection during exploitation."""
        # Setup for exploitation
        mock_rand.return_value = 0.5  # Above epsilon=0.3, should exploit
        
        # Create a test state
        state = np.array([30, 10, 5, 2, 1, 0])
        
        # Mock the model to return specific Q-values
        self.agent.model = MagicMock()
        mock_q_values = torch.tensor([[0.2, 0.1, 0.4, 0.3]])  # Action 2 has highest value
        self.agent.model.return_value = mock_q_values
        
        # Choose action
        action = self.agent.choose_action(state)
        
        # Verify exploitation
        self.agent.model.assert_called_once()
        self.assertEqual(action, 2)

    def test_remember(self):
        """Test storing experiences in memory."""
        # Create sample experience
        state = np.array([30, 10, 5, 2, 1, 0])
        action = 2
        reward = 0.5
        next_state = np.array([25, 8, 4, 3, 0, 1])
        done = False
        
        # Store the experience
        self.agent.remember(state, action, reward, next_state, done)
        
        # Verify memory contains one experience
        self.assertEqual(len(self.agent.memory), 1)
        
        # Verify the stored experience
        stored_exp = self.agent.memory[0]
        np.testing.assert_array_equal(stored_exp[0], state)
        self.assertEqual(stored_exp[1], 2)  # Action index
        self.assertEqual(stored_exp[2], reward)
        np.testing.assert_array_equal(stored_exp[3], next_state)
        self.assertEqual(stored_exp[4], done)

    def test_learn_no_replay(self):
        """Test learning when memory size is below batch size."""
        # Create sample experience
        state = np.array([30, 10, 5, 2, 1, 0])
        action = 2
        next_state = np.array([25, 8, 4, 3, 0, 1])
        done = False
        
        # Mock reward calculation
        self.agent.calculate_reward = MagicMock(return_value=(0.5, {}))
        
        # Mock replay method to verify it's not called
        self.agent._replay = MagicMock()
        
        # Learn from the experience
        self.agent.learn(state, action, next_state, done)
        
        # Verify remember was called but not replay
        self.assertEqual(len(self.agent.memory), 1)
        self.agent._replay.assert_not_called()

    @patch('random.sample')
    def test_replay(self, mock_sample):
        """Test experience replay and network updates."""
        # Fill memory with enough experiences
        for i in range(20):  # More than batch_size=16
            state = np.array([30+i, 10, 5, 2, 1, 0])
            next_state = np.array([25+i, 8, 4, 3, 0, 1])
            self.agent.memory.append((state, i % 4, 0.5, next_state, False))
        
        # Mock random sample
        mock_sample.return_value = [self.agent.memory[i] for i in range(16)]
        
        # Create a new simple model and optimizer that won't have gradient issues
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(6, 4, bias=False)
                # Initialize with ones for predictable output
                torch.nn.init.ones_(self.layer.weight)
                
            def forward(self, x):
                return self.layer(x)
                
        # Replace model and target model with our simple version
        self.agent.model = SimpleModel()
        self.agent.target_model = SimpleModel()
        
        # Create a real optimizer instead of a mock
        self.agent.optimizer = torch.optim.SGD(self.agent.model.parameters(), lr=0.01)
        
        # Get initial epsilon for comparison
        initial_epsilon = self.agent.epsilon
        
        # Run replay
        self.agent._replay()
        
        # Verify epsilon decay
        self.assertLess(self.agent.epsilon, initial_epsilon)  # Should be reduced
        
        # Verify train step increment
        self.assertEqual(self.agent.train_step, 1)

    def test_create_method(self):
        """Test the create class method."""
        # Call create with custom parameters
        result = DQNAgent.create('tls1', self.mock_network, epsilon=0.5, batch_size=64)
        
        # Verify agent was created with correct parameters
        self.assertIsInstance(result, DQNAgent)
        self.assertEqual(result.tls_id, 'tls1')
        self.assertEqual(result.network, self.mock_network)
        self.assertEqual(result.epsilon, 0.5)
        self.assertEqual(result.batch_size, 64)


class TestDQN(unittest.TestCase):
    """Test case for the DQN neural network model."""

    def test_initialization(self):
        """Test DQN network initialization."""
        network = DQN(state_size=6, action_size=4)
        
        # Verify layer dimensions
        self.assertEqual(network.fc1.in_features, 6)
        self.assertEqual(network.fc1.out_features, 24)
        self.assertEqual(network.fc2.in_features, 24)
        self.assertEqual(network.fc2.out_features, 24)
        self.assertEqual(network.fc3.in_features, 24)
        self.assertEqual(network.fc3.out_features, 4)

    def test_forward_pass(self):
        """Test forward pass through the network."""
        network = DQN(state_size=6, action_size=4)
        
        # Create sample input
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=torch.float32)
        
        # Forward pass
        output = network(x)
        
        # Verify output dimensions
        self.assertEqual(output.shape, torch.Size([1, 4]))


if __name__ == '__main__':
    unittest.main()