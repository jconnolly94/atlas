"""Integration tests for data collection and observer pattern."""
import unittest
from unittest.mock import MagicMock, patch
import os
import tempfile
import shutil

from src.utils.data_collector import DataCollector
from src.env.environment import Environment
from src.env.network import Network


class TestDataCollectionIntegration(unittest.TestCase):
    """Test data collection integration with environment."""

    def setUp(self):
        """Set up test environment with data collector."""
        # Create a temporary directory for data storage
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock network
        self.mock_network = MagicMock(spec=Network)
        self.mock_network.tls_ids = ['tl1', 'tl2']
        
        # Create environment with mocked network
        self.env = Environment(self.mock_network)
        self.env.episode_number = 1
        self.env.episode_step = 1
        
        # Create data collector
        self.data_collector = DataCollector(agent_type="test_agent", network_name="test_network")
        
        # Register data collector as observer
        self.env.add_observer(self.data_collector)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_data_collector_registers_with_environment(self):
        """Test that data collector registers as observer with environment."""
        # Verify data collector is in environment's observers list
        self.assertIn(self.data_collector, self.env._observers)
        
    def test_environment_notifies_data_collector_on_step(self):
        """Test that environment notifies data collector when simulation steps."""
        # Patch data collector's on_step_complete method
        with patch.object(self.data_collector, 'on_step_complete') as mock_on_step:
            # Create step data
            step_data = {
                'episode': self.env.episode_number,
                'step': self.env.episode_step,
                'tls_id': 'tl1',
                'action': 0,
                'reward': 10.0
            }
            # Notify observers
            self.env.notify_step_complete(step_data)
            
            # Verify on_step_complete was called with data
            mock_on_step.assert_called_once_with(step_data)
            
    def test_environment_notifies_data_collector_on_episode_end(self):
        """Test that environment notifies data collector when episode ends."""
        # Patch data collector's on_episode_complete method
        with patch.object(self.data_collector, 'on_episode_complete') as mock_on_episode_complete:
            # Create episode data
            episode_data = {
                'episode': self.env.episode_number,
                'total_steps': 10,
                'avg_waiting': 5.0,
                'total_reward': 50.0
            }
            
            # Notify observers
            self.env.notify_episode_complete(episode_data)
            
            # Verify on_episode_complete was called with data
            mock_on_episode_complete.assert_called_once_with(episode_data)
            
    def test_data_collector_records_step_data(self):
        """Test that data collector records step data."""
        # Setup step data
        step_data = {
            'episode': self.env.episode_number,
            'step': self.env.episode_step,
            'tls_id': 'tl1',
            'action': 0,
            'reward': 10.0,
            'waiting_time': 5.0,
            'queue_length': 2
        }
        
        # Record step data
        self.data_collector.on_step_complete(step_data)
        
        # Verify step data was collected
        self.assertEqual(len(self.data_collector.step_data), 1)
        
        # Verify step data contains expected fields
        recorded_data = self.data_collector.step_data[0]
        self.assertEqual(recorded_data['episode'], 1)
        self.assertEqual(recorded_data['step'], 1)
        self.assertEqual(recorded_data['tls_id'], 'tl1')
        self.assertEqual(recorded_data['reward'], 10.0)
        
    def test_data_collector_records_episode_data(self):
        """Test that data collector records episode data."""
        # Setup episode data
        episode_data = {
            'episode': self.env.episode_number,
            'steps': 10,
            'avg_waiting_time': 5.0,
            'total_reward': 50.0
        }
        
        # Record episode data
        self.data_collector.on_episode_complete(episode_data)
        
        # Verify episode data was collected
        self.assertEqual(len(self.data_collector.episode_data), 1)
        
        # Verify episode data contains expected fields
        recorded_data = self.data_collector.episode_data[0]
        self.assertEqual(recorded_data['episode'], 1)
        self.assertEqual(recorded_data['steps'], 10)
        self.assertEqual(recorded_data['avg_waiting_time'], 5.0)
        self.assertEqual(recorded_data['total_reward'], 50.0)
        
    def test_data_collector_saves_data_to_files(self):
        """Test that data collector saves data to files."""
        # Make sure data directories exist
        os.makedirs('data/datastore/steps', exist_ok=True)
        os.makedirs('data/datastore/episodes', exist_ok=True)
        
        # Setup basic step data
        self.data_collector.step_data = [
            {'episode': 1, 'step': 1, 'tls_id': 'tl1', 'reward': 10.0, 'agent_type': 'test_agent', 'network': 'test_network'},
            {'episode': 1, 'step': 2, 'tls_id': 'tl1', 'reward': 8.0, 'agent_type': 'test_agent', 'network': 'test_network'}
        ]
        
        # Setup basic episode data
        self.data_collector.episode_data = [
            {'episode': 1, 'steps': 10, 'avg_waiting_time': 5.0, 'total_reward': 50.0, 'agent_type': 'test_agent', 'network': 'test_network'}
        ]
        
        # Save data
        self.data_collector.save_data("test")
        
        # Verify step data file was created
        step_file = 'data/datastore/steps/step_data_test.csv'
        self.assertTrue(os.path.exists(step_file))
        
        # Verify episode data file was created
        episode_file = 'data/datastore/episodes/episode_data_test.csv'
        self.assertTrue(os.path.exists(episode_file))
        
        # Clean up test files
        os.remove(step_file)
        os.remove(episode_file)


if __name__ == '__main__':
    unittest.main()