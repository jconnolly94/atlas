import unittest
from unittest.mock import MagicMock, patch, mock_open
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from src.utils.data_collector import DataCollector, NullDataCollector, MetricsCalculator


class TestDataCollector(unittest.TestCase):
    """Test case for the DataCollector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.collector = DataCollector(agent_type="DQN", network_name="TestNetwork")
        
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create necessary directories
        os.makedirs('data/datastore/steps', exist_ok=True)
        os.makedirs('data/datastore/episodes', exist_ok=True)

    def tearDown(self):
        """Clean up after tests."""
        # Change back to original directory and remove temp directory
        os.chdir(self.original_dir)
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test DataCollector initialization."""
        self.assertEqual(self.collector.agent_type, "DQN")
        self.assertEqual(self.collector.network_name, "TestNetwork")
        self.assertEqual(self.collector.step_data, [])
        self.assertEqual(self.collector.episode_data, [])
    
    def test_on_step_complete(self):
        """Test recording step data."""
        # Create test data with NumPy values
        step_data = {
            'step': 1,
            'tls_id': 'tls1',
            'reward': np.float32(0.5),
            'waiting_time': np.float64(20.5),
            'vehicle_count': np.int32(10)
        }
        
        # Record the data
        self.collector.on_step_complete(step_data)
        
        # Verify data was recorded and NumPy types were converted
        self.assertEqual(len(self.collector.step_data), 1)
        recorded = self.collector.step_data[0]
        
        self.assertEqual(recorded['agent_type'], "DQN")
        self.assertEqual(recorded['network'], "TestNetwork")
        self.assertEqual(recorded['step'], 1)
        self.assertEqual(recorded['tls_id'], 'tls1')
        
        # Check NumPy types were converted
        self.assertIsInstance(recorded['reward'], float)
        self.assertIsInstance(recorded['waiting_time'], float)
        self.assertIsInstance(recorded['vehicle_count'], int)
        
        # Check values were preserved
        self.assertAlmostEqual(recorded['reward'], 0.5)
        self.assertAlmostEqual(recorded['waiting_time'], 20.5)
        self.assertEqual(recorded['vehicle_count'], 10)
    
    def test_on_episode_complete(self):
        """Test recording episode data."""
        # Create test data with NumPy values
        episode_data = {
            'episode': 1,
            'total_reward': np.float32(100.5),
            'avg_waiting_time': np.float64(15.5),
            'total_vehicle_count': np.int32(50)
        }
        
        # Record the data
        self.collector.on_episode_complete(episode_data)
        
        # Verify data was recorded and NumPy types were converted
        self.assertEqual(len(self.collector.episode_data), 1)
        recorded = self.collector.episode_data[0]
        
        self.assertEqual(recorded['agent_type'], "DQN")
        self.assertEqual(recorded['network'], "TestNetwork")
        self.assertEqual(recorded['episode'], 1)
        
        # Check NumPy types were converted
        self.assertIsInstance(recorded['total_reward'], float)
        self.assertIsInstance(recorded['avg_waiting_time'], float)
        self.assertIsInstance(recorded['total_vehicle_count'], int)
        
        # Check values were preserved
        self.assertAlmostEqual(recorded['total_reward'], 100.5)
        self.assertAlmostEqual(recorded['avg_waiting_time'], 15.5)
        self.assertEqual(recorded['total_vehicle_count'], 50)
    
    @patch('pandas.DataFrame.to_csv')
    def test_save_data(self, mock_to_csv):
        """Test saving data to CSV files."""
        # Add some data
        self.collector.step_data = [{'step': 1, 'value': 10}]
        self.collector.episode_data = [{'episode': 1, 'total_reward': 100}]
        
        # Save data
        self.collector.save_data("test_suffix")
        
        # Verify to_csv was called twice (once for steps, once for episodes)
        self.assertEqual(mock_to_csv.call_count, 2)
        
        # Verify file paths
        call_args_list = [call[0][0] for call in mock_to_csv.call_args_list]
        self.assertIn('data/datastore/steps/step_data_test_suffix.csv', call_args_list)
        self.assertIn('data/datastore/episodes/episode_data_test_suffix.csv', call_args_list)
    
    @patch('pandas.DataFrame.to_csv')
    def test_save_data_empty(self, mock_to_csv):
        """Test saving when no data is available."""
        # Save data when empty
        self.collector.save_data("test_suffix")
        
        # Verify to_csv was not called
        mock_to_csv.assert_not_called()
    
    @patch('pandas.concat')
    @patch('pandas.read_csv')
    @patch('glob.glob')
    def test_combine_results(self, mock_glob, mock_read_csv, mock_concat):
        """Test combining results from multiple files."""
        # Setup mock files
        mock_glob.side_effect = lambda pattern: {
            'data/datastore/steps/step_data_*.csv': ['data/datastore/steps/step_data_1.csv', 
                                                   'data/datastore/steps/step_data_2.csv'],
            'data/datastore/episodes/episode_data_*.csv': ['data/datastore/episodes/episode_data_1.csv', 
                                                        'data/datastore/episodes/episode_data_2.csv']
        }[pattern]
        
        # Setup mock dataframes
        df1 = pd.DataFrame({'step': [1, 2], 'value': [10, 20]})
        df2 = pd.DataFrame({'step': [3, 4], 'value': [30, 40]})
        mock_read_csv.return_value = df1  # Simplified, assume all files have same content
        
        # Mock concat to return a combined dataframe with to_csv method
        combined_df = pd.DataFrame({'step': [1, 2, 3, 4], 'value': [10, 20, 30, 40]})
        combined_df.to_csv = MagicMock()
        mock_concat.return_value = combined_df
        
        # Call combine_results
        with patch('os.remove') as mock_remove:
            DataCollector.combine_results()
            
            # Verify glob was called
            self.assertEqual(mock_glob.call_count, 2)
            
            # Verify read_csv was called 4 times (2 step files, 2 episode files)
            self.assertEqual(mock_read_csv.call_count, 4)
            
            # Verify concat was called 2 times (once for steps, once for episodes)
            self.assertEqual(mock_concat.call_count, 2)
            
            # Verify to_csv was called 2 times (once for steps, once for episodes)
            self.assertEqual(combined_df.to_csv.call_count, 2)
            
            # Verify os.remove was called 4 times (to clean up original files)
            self.assertEqual(mock_remove.call_count, 4)


class TestNullDataCollector(unittest.TestCase):
    """Test case for the NullDataCollector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.collector = NullDataCollector()
    
    def test_no_op_methods(self):
        """Test that methods don't do anything."""
        # These method calls should not raise any exceptions
        self.collector.on_step_complete({'step': 1})
        self.collector.on_episode_complete({'episode': 1})
        self.collector.save_data("test")
        
        # No assertions needed as the methods should just do nothing


class TestMetricsCalculator(unittest.TestCase):
    """Test case for the MetricsCalculator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_network = MagicMock()
        self.mock_network.get_controlled_lanes.return_value = ['lane1', 'lane2']
        self.mock_network.get_lane_waiting_time.side_effect = lambda lane: 30 if lane == 'lane1' else 20
        self.mock_network.get_lane_vehicle_count.side_effect = lambda lane: 5 if lane == 'lane1' else 3
        self.mock_network.get_lane_queue.side_effect = lambda lane: 2 if lane == 'lane1' else 1
        self.mock_network.get_phase_duration.return_value = 30
        
        self.calculator = MetricsCalculator(self.mock_network)
    
    def test_calculate_step_metrics(self):
        """Test calculating metrics for a step."""
        # Calculate metrics
        metrics = self.calculator.calculate_step_metrics('tls1', 0, 0.5)
        
        # Verify metrics
        self.assertEqual(metrics['tls_id'], 'tls1')
        self.assertEqual(metrics['action'], 0)
        self.assertEqual(metrics['reward'], 0.5)
        self.assertEqual(metrics['waiting_time'], 50)  # 30 + 20
        self.assertEqual(metrics['vehicle_count'], 8)  # 5 + 3
        self.assertEqual(metrics['queue_length'], 3)   # 2 + 1
        self.assertEqual(metrics['phase_duration'], 30)
    
    def test_calculate_step_metrics_exception(self):
        """Test calculating metrics when an exception occurs."""
        # Configure mock to raise exception
        self.mock_network.get_controlled_lanes.side_effect = Exception("Network error")
        
        # Calculate metrics
        metrics = self.calculator.calculate_step_metrics('tls1', 0, 0.5)
        
        # Verify default metrics are returned
        self.assertEqual(metrics['tls_id'], 'tls1')
        self.assertEqual(metrics['action'], 0)
        self.assertEqual(metrics['reward'], 0.5)
        self.assertEqual(metrics['waiting_time'], 0.0)
        self.assertEqual(metrics['vehicle_count'], 0)
        self.assertEqual(metrics['queue_length'], 0)
        self.assertEqual(metrics['phase_duration'], 0.0)


if __name__ == '__main__':
    unittest.main()