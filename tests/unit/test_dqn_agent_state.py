import unittest
import os
import tempfile
import shutil
import torch
import json
from unittest.mock import MagicMock, patch
import numpy as np

from src.agents.dqn_agent import DQNAgent, TrafficNetworkDQN


class TestDQNAgentState(unittest.TestCase):
    """Test case for the DQNAgent state saving and loading functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock network for testing
        self.mock_network = MagicMock()
        self.mock_network.get_departed_vehicles_count.return_value = 10
        
        # Create a test directory for saving/loading
        self.test_dir = tempfile.mkdtemp()
        
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

    def tearDown(self):
        """Clean up after tests."""
        # Remove the test directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_save_state(self):
        """Test saving agent state to disk."""
        # Modify some agent parameters to ensure they're saved
        self.agent.epsilon = 0.25
        self.agent.train_step = 100
        
        # Save state
        self.agent.save_state(self.test_dir)
        
        # Check if files were created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'model.pth')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'target_model.pth')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'optimizer.pth')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'hyperparams.json')))
        
        # Check content of hyperparameters file
        with open(os.path.join(self.test_dir, 'hyperparams.json'), 'r') as f:
            params = json.load(f)
            self.assertEqual(params['epsilon'], 0.25)
            self.assertEqual(params['gamma'], 0.9)
            self.assertEqual(params['train_step'], 100)
            self.assertEqual(params['epsilon_decay'], 0.99)
            self.assertEqual(params['epsilon_min'], 0.05)
            self.assertEqual(params['batch_size'], 16)
            self.assertEqual(params['memory_size'], 1000)
            self.assertEqual(params['target_update_freq'], 50)

    def test_load_state(self):
        """Test loading agent state from disk."""
        # Modify and save agent state
        original_epsilon = 0.25
        original_train_step = 100
        self.agent.epsilon = original_epsilon
        self.agent.train_step = original_train_step
        
        # Get original model parameters for comparison
        original_model_params = self.agent.model.state_dict()
        original_target_model_params = self.agent.target_model.state_dict()
        
        # Save state
        self.agent.save_state(self.test_dir)
        
        # Create a new agent with different parameters
        new_agent = DQNAgent(
            'tls1', 
            self.mock_network, 
            alpha=0.02,  # Different learning rate
            gamma=0.8,   # Different discount factor
            epsilon=0.5, # Different exploration rate
            epsilon_decay=0.999,
            epsilon_min=0.1,
            batch_size=32,
            memory_size=2000,
            target_update_freq=100
        )
        
        # Load state from first agent
        new_agent.load_state(self.test_dir)
        
        # Check if hyperparameters were restored correctly
        self.assertEqual(new_agent.epsilon, original_epsilon)
        self.assertEqual(new_agent.train_step, original_train_step)
        self.assertEqual(new_agent.gamma, 0.9)
        self.assertEqual(new_agent.epsilon_decay, 0.99)
        self.assertEqual(new_agent.epsilon_min, 0.05)
        
        # Check if model parameters were restored correctly
        for key in original_model_params:
            self.assertTrue(torch.allclose(
                original_model_params[key], 
                new_agent.model.state_dict()[key]
            ))
            
        # Check if target model parameters were restored correctly
        for key in original_target_model_params:
            self.assertTrue(torch.allclose(
                original_target_model_params[key], 
                new_agent.target_model.state_dict()[key]
            ))

    def test_load_state_missing_files(self):
        """Test handling of missing files during load."""
        # Create a directory without state files
        empty_dir = os.path.join(self.test_dir, 'empty')
        os.makedirs(empty_dir, exist_ok=True)
        
        # Try to load from empty directory
        original_epsilon = self.agent.epsilon
        self.agent.load_state(empty_dir)
        
        # Ensure agent state remains unchanged
        self.assertEqual(self.agent.epsilon, original_epsilon)

    def test_load_state_device_compatibility(self):
        """Test loading state across different devices (CPU)."""
        # Save state
        self.agent.save_state(self.test_dir)
        
        # Create a new agent
        new_agent = DQNAgent('tls1', self.mock_network)
        
        # Mock torch.cuda.is_available to ensure device is set to CPU
        with patch('torch.cuda.is_available', return_value=False):
            new_agent.load_state(self.test_dir)
            
        # Check model device
        for param in new_agent.model.parameters():
            self.assertEqual(param.device.type, 'cpu')
            
        # Check target model device
        for param in new_agent.target_model.parameters():
            self.assertEqual(param.device.type, 'cpu')

    def test_save_state_error_handling(self):
        """Test error handling during save state."""
        # Create invalid directory (file instead of directory)
        invalid_dir = os.path.join(self.test_dir, 'invalid_file')
        with open(invalid_dir, 'w') as f:
            f.write('test')
            
        # Try to save to invalid location
        # This shouldn't raise an exception, just log the error
        self.agent.save_state(invalid_dir)
        
        # Check that no files were created in parent directory
        parent_files = os.listdir(self.test_dir)
        self.assertEqual(len(parent_files), 1)  # Just the invalid_file
        

if __name__ == '__main__':
    unittest.main()
