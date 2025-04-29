"""Integration tests for agent and environment interaction."""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.agents.agent import Agent
from src.agents.q_agent import QAgent
from src.env.environment import Environment
from src.env.network import Network


class TestAgentEnvironmentIntegration(unittest.TestCase):
    """Test interactions between agent and environment."""

    def setUp(self):
        """Set up test environment with mocked network."""
        # Create mock network
        self.mock_network = MagicMock(spec=Network)
        
        # Setup network mock
        self.mock_network.tls_ids = ['tl1', 'tl2']
        self.mock_network.simulation_steps_per_action = 5
        self.mock_network.get_possible_phases.return_value = [0, 1]
        
        # We need to set possible_actions directly since it's set in __init__
        # before we can patch the method
        self.mock_network.get_controlled_lanes.return_value = ['lane1', 'lane2']
        self.mock_network.get_adjacent_traffic_lights.return_value = ['tl2']
        self.mock_network.get_lane_waiting_time.return_value = 10
        self.mock_network.get_lane_queue.return_value = 3
        
        # Mock get_state to return realistic state vectors
        mock_states = {
            'tl1': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            'tl2': np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        }
        self.mock_network.get_state.return_value = mock_states
        
        # Create environment with mocked network
        self.env = Environment(self.mock_network)
        
        # Create agent with default config
        self.agent_config = {
            'alpha': 0.1,           # Learning rate
            'gamma': 0.95,          # Discount factor
            'epsilon': 0.1,         # Exploration rate
            'state_bin_size': 5     # State bin size
        }
        self.agent = QAgent('tl1', self.mock_network, **self.agent_config)
        
        # Set possible_actions directly for testing
        self.agent.possible_actions = [0, 1]

    def test_agent_chooses_action_based_on_environment_state(self):
        """Test that agent can choose an action based on environment state."""
        # Reset environment to get initial state
        self.env.reset()
        
        # Get state for agent from mock_states
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        
        # Agent should be able to choose an action based on this state
        action = self.agent.choose_action(state)
        
        # Verify action is valid (in range of possible traffic light phases)
        self.assertIn(action, [0, 1])
        
    def test_agent_learns_from_environment_feedback(self):
        """Test that agent can learn from environment feedback."""
        # Get initial state
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        
        # Choose action
        action = 0
        
        # Get new state
        new_state = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        
        # Done flag
        done = False
        
        # Learn from experience
        self.agent.learn(state, action, new_state, done)
        
        # Verify agent has Q-values storage
        self.assertTrue(hasattr(self.agent, 'q_values'))
        
    def test_environment_applies_agent_actions(self):
        """Test that environment applies agent actions to the network."""
        # Create a dict of agents
        agents = {'tl1': self.agent}
        
        # Add state to environment's current_states
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.env.current_states = {'tl1': state}
        
        # Mock agent's methods
        self.agent.choose_action = MagicMock(return_value=1)
        
        # Mock the calculate_reward method
        self.agent.calculate_reward = MagicMock(return_value=(10.0, {}))
        
        # Step environment
        self.env.step(agents)
        
        # Since we mocked the agent to return action 1, verify it was passed to network
        self.mock_network.set_traffic_light_phase.assert_called_with('tl1', 1)
    
    def test_environment_provides_reward_to_agent(self):
        """Test that environment provides reward feedback to agent."""
        # Create a dict of agents
        agents = {'tl1': self.agent}
        
        # Add state to environment's current_states
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.env.current_states = {'tl1': state}
        
        # Mock get_state to return next state
        next_state = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.mock_network.get_state.return_value = {'tl1': next_state}
        
        # Setup agent to return reward components
        self.agent.choose_action = MagicMock(return_value=0)
        self.agent.calculate_reward = MagicMock(return_value=(10.0, {'waiting_time': 5.0}))
        
        # Patch agent's learn method
        with patch.object(self.agent, 'learn') as mock_learn:
            # Step environment
            self.env.step(agents)
            
            # Verify agent's learn method was called
            mock_learn.assert_called()
            
            # Verify the first three args match what we expect
            call_args = mock_learn.call_args[0]
            self.assertTrue(np.array_equal(call_args[0], state))
            self.assertEqual(call_args[1], 0)
            self.assertTrue(np.array_equal(call_args[2], next_state))
            # The 4th arg (done) might be a mock, just check it exists
            self.assertEqual(len(call_args), 4)


if __name__ == '__main__':
    unittest.main()