"""Integration tests for environment and network interactions."""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.env.environment import Environment
from src.env.network import Network


class TestEnvironmentNetworkIntegration(unittest.TestCase):
    """Test interactions between environment and network."""

    def setUp(self):
        """Set up environment with mocked network."""
        # Create a mocked network instead of using real TraCI patches
        self.network = MagicMock(spec=Network)
        
        # Configure the mock network
        self.network.tls_ids = ['tl1', 'tl2']
        self.network.simulation_steps_per_action = 5
        
        # Mock get_state to return realistic state vectors
        mock_states = {
            'tl1': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            'tl2': np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        }
        self.network.get_state.return_value = mock_states
        
        # Create environment with mocked network
        self.env = Environment(self.network)

    def test_environment_initialization_with_network(self):
        """Test that environment initializes correctly with network."""
        # Verify environment has reference to network
        self.assertEqual(self.env.network, self.network)
        
        # Verify environment sets up initial state
        self.assertEqual(self.env.episode_number, 0)
        self.assertEqual(self.env.episode_step, 0)
            
    def test_environment_reset_calls_network_reset(self):
        """Test that environment reset calls network reset."""
        # Reset environment
        self.env.reset()
        
        # Verify network reset was called
        self.network.reset_simulation.assert_called_once()
        
        # Verify environment state was reset
        self.assertEqual(self.env.episode_number, 1)
        self.assertEqual(self.env.episode_step, 0)
            
    def test_environment_step_advances_network(self):
        """Test that environment step advances network simulation."""
        # Create mock agents
        mock_agents = {
            'tl1': MagicMock(),
            'tl2': MagicMock()
        }
        
        # Configure mock agent behavior
        for agent in mock_agents.values():
            agent.choose_action.return_value = 0
            agent.calculate_reward.return_value = (10.0, {})
        
        # Step environment with agents
        self.env.step(mock_agents)
        
        # Verify network simulation was advanced
        self.assertTrue(self.network.simulation_step.called)
            
    def test_environment_gets_state_from_network(self):
        """Test that environment retrieves state information from network."""
        # Get state from environment
        state = self.env.get_state()
        
        # Verify network method was called
        self.network.get_state.assert_called_once()
        
        # Verify state structure
        self.assertEqual(len(state), 2)  # Two traffic lights
        self.assertIn('tl1', state)
        self.assertIn('tl2', state)
                    
    def test_environment_sets_traffic_light_phase_on_network(self):
        """Test that environment sets traffic light phases on network."""
        # Setup actions dict
        actions = {'tl1': 1, 'tl2': 2}
        
        # Apply actions
        self.env.apply_actions(actions)
        
        # Verify network methods were called with correct args
        self.network.set_traffic_light_phase.assert_any_call('tl1', 1)
        self.network.set_traffic_light_phase.assert_any_call('tl2', 2)
            
    def test_environment_terminates_when_network_indicates_simulation_end(self):
        """Test that environment terminates when network indicates simulation end."""
        # First, network reports simulation not done
        self.network.is_simulation_complete.return_value = False
        self.assertFalse(self.env.is_terminal_state())
            
        # Then, network reports simulation is done
        self.network.is_simulation_complete.return_value = True
        self.assertTrue(self.env.is_terminal_state())
            

if __name__ == '__main__':
    unittest.main()