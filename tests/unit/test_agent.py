import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from src.agents.agent import Agent


class TestAgent(unittest.TestCase):
    """Test case for the Agent base class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock network for testing
        self.mock_network = MagicMock()
        self.mock_network.get_controlled_lanes.return_value = ['lane1', 'lane2']
        self.mock_network.get_lane_queue.side_effect = lambda lane: 5 if lane == 'lane1' else 3
        self.mock_network.get_lane_waiting_time.side_effect = lambda lane: 30 if lane == 'lane1' else 20
        self.mock_network.get_lane_vehicle_count.side_effect = lambda lane: 8 if lane == 'lane1' else 5
        self.mock_network.get_adjacent_traffic_lights.return_value = ['tls2', 'tls3']
        
        # Create a concrete subclass of Agent for testing
        class ConcreteAgent(Agent):
            def choose_action(self, state):
                return 0
                
            def learn(self, state, action, next_state, done):
                pass
                
        # Create an instance of the concrete agent
        self.agent = ConcreteAgent('tls1', self.mock_network)

    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.tls_id, 'tls1')
        self.assertEqual(self.agent.network, self.mock_network)

    def test_calculate_reward(self):
        """Test reward calculation."""
        # Create sample state data
        state = np.array([0, 0, 0, 0])  # Initial state doesn't matter for this test
        next_state = np.array([50, 13, 8, 5])  # waiting_time, vehicle_count, queue_length, throughput
        action = 0
        
        # Calculate reward
        reward, components = self.agent.calculate_reward(state, action, next_state)
        
        # Verify reward calculation (approximate check due to normalization)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(components, dict)
        
        # Verify reward components
        self.assertIn('waiting_penalty', components)
        self.assertIn('throughput_reward', components)
        self.assertIn('balance_penalty', components)
        self.assertIn('change_penalty', components)
        self.assertIn('congestion_penalty', components)
        
        # Check specific component values
        self.assertLess(components['waiting_penalty'], 0)  # Should be negative
        self.assertGreater(components['throughput_reward'], 0)  # Should be positive

    def test_get_adjacent_traffic_lights(self):
        """Test getting adjacent traffic lights."""
        adjacent_tls = self.agent.get_adjacent_traffic_lights(self.mock_network)
        self.assertEqual(adjacent_tls, ['tls2', 'tls3'])
        self.mock_network.get_adjacent_traffic_lights.assert_called_once_with('tls1')

    @patch('src.agents.agent.Agent.DEFAULT_CONFIG', {'param1': 'value1'})
    def test_create_class_method(self):
        """Test the create class method."""
        # Setup mock class
        mock_cls = MagicMock()
        mock_cls.DEFAULT_CONFIG = {'param1': 'value1'}
        mock_cls.return_value = 'agent_instance'
        
        # Call the create method
        result = Agent.create.__func__(mock_cls, 'tls1', self.mock_network, param2='value2')
        
        # Verify result
        self.assertEqual(result, 'agent_instance')
        
        # Verify the class was instantiated with correct parameters
        expected_config = {'param1': 'value1', 'param2': 'value2'}
        mock_cls.assert_called_once_with('tls1', self.mock_network, **expected_config)


if __name__ == '__main__':
    unittest.main()