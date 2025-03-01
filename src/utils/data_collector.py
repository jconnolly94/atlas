import pandas as pd
import os
import glob
import threading
import time
from typing import Dict, List, Any, Optional
import logging

from .observer import Observer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='data_collector.log'
)
logger = logging.getLogger('DataCollector')


class DataCollector(Observer):
    """Collects and stores simulation data using the observer pattern."""

    def __init__(self, agent_type: str, network_name: str):
        self.agent_type = agent_type
        self.network_name = network_name
        self.step_data: List[Dict[str, Any]] = []
        self.episode_data: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        logger.info(f"DataCollector initialized for {agent_type} on {network_name}")

    def on_step_complete(self, data: Dict[str, Any]) -> None:
        """Record data from a completed simulation step."""
        with self.lock:
            # Add common fields to the data
            record = {
                'agent_type': self.agent_type,
                'network': self.network_name,
                **data
            }

            # Ensure all values are Python native types (not NumPy types)
            for key, value in record.items():
                if hasattr(value, 'item'):  # Check if it's a NumPy scalar
                    try:
                        record[key] = value.item()  # Convert to Python native type
                    except:
                        record[key] = float(value)  # Fallback to float

            self.step_data.append(record)

    def on_episode_complete(self, data: Dict[str, Any]) -> None:
        """Record data from a completed episode."""
        with self.lock:
            # Add common fields to the data
            record = {
                'agent_type': self.agent_type,
                'network': self.network_name,
                **data
            }

            # Ensure all values are Python native types (not NumPy types)
            for key, value in record.items():
                if hasattr(value, 'item'):  # Check if it's a NumPy scalar
                    try:
                        record[key] = value.item()  # Convert to Python native type
                    except:
                        record[key] = float(value)  # Fallback to float

            self.episode_data.append(record)
            logger.info(f"Episode complete: {record}")

    def save_data(self, suffix: str) -> None:
        """Save collected data to CSV files."""
        with self.lock:
            try:
                # Create directories if they don't exist
                os.makedirs('data/datastore/steps', exist_ok=True)
                os.makedirs('data/datastore/episodes', exist_ok=True)

                step_file = f'data/datastore/steps/step_data_{suffix}.csv'
                episode_file = f'data/datastore/episodes/episode_data_{suffix}.csv'

                # Check if we have data to save
                if not self.step_data:
                    logger.warning(f"No step data to save for {suffix}")
                    # Create a minimal dataset to avoid empty file errors
                    self.step_data.append({
                        'agent_type': self.agent_type,
                        'network': self.network_name,
                        'episode': 0,
                        'step': 0,
                        'tls_id': 'none',
                        'action': 0,
                        'reward': 0.0,
                        'waiting_time': 0.0,
                        'vehicle_count': 0,
                        'queue_length': 0,
                        'phase_duration': 0.0
                    })

                if not self.episode_data:
                    logger.warning(f"No episode data to save for {suffix}")
                    # Create a minimal dataset to avoid empty file errors
                    self.episode_data.append({
                        'agent_type': self.agent_type,
                        'network': self.network_name,
                        'episode': 0,
                        'avg_waiting': 0.0,
                        'arrived_vehicles': 0,
                        'total_reward': 0.0
                    })

                # Convert to dataframes and save
                step_df = pd.DataFrame(self.step_data)
                episode_df = pd.DataFrame(self.episode_data)

                step_df.to_csv(step_file, index=False)
                episode_df.to_csv(episode_file, index=False)

                logger.info(f"Data saved successfully: {step_file} and {episode_file}")
            except Exception as e:
                logger.error(f"Error saving data: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())

    @staticmethod
    def combine_results() -> None:
        """Combine results from multiple data collectors."""
        try:
            # Wait a moment to ensure all files are fully written
            time.sleep(1)

            # Check if directories exist
            os.makedirs('data/datastore/steps', exist_ok=True)
            os.makedirs('data/datastore/episodes', exist_ok=True)

            # Combine step data
            step_files = glob.glob('data/datastore/steps/step_data_*.csv')
            if step_files:
                logger.info(f"Found {len(step_files)} step data files to combine")

                # Check if files are valid
                valid_step_dfs = []
                for f in step_files:
                    try:
                        df = pd.read_csv(f)
                        if not df.empty:
                            valid_step_dfs.append(df)
                        else:
                            logger.warning(f"Empty step data file: {f}")
                    except Exception as e:
                        logger.error(f"Error reading step file {f}: {str(e)}")

                if valid_step_dfs:
                    combined_steps = pd.concat(valid_step_dfs)
                    combined_steps.to_csv('data/datastore/steps/combined_step_data.csv', index=False)
                    logger.info(f"Combined {len(valid_step_dfs)} step data files successfully")

                    # Cleanup after combining
                    for f in step_files:
                        try:
                            os.remove(f)
                        except Exception as e:
                            logger.error(f"Error removing file {f}: {str(e)}")
                else:
                    logger.warning("No valid step data files to combine")
                    # Create an empty combined file to avoid errors
                    pd.DataFrame(columns=[
                        'agent_type', 'network', 'episode', 'step', 'tls_id',
                        'action', 'reward', 'waiting_time', 'vehicle_count',
                        'queue_length', 'phase_duration'
                    ]).to_csv('data/datastore/steps/combined_step_data.csv', index=False)
            else:
                logger.warning("No step data files found")
                # Create an empty combined file to avoid errors
                pd.DataFrame(columns=[
                    'agent_type', 'network', 'episode', 'step', 'tls_id',
                    'action', 'reward', 'waiting_time', 'vehicle_count',
                    'queue_length', 'phase_duration'
                ]).to_csv('data/datastore/steps/combined_step_data.csv', index=False)

            # Combine episode data
            episode_files = glob.glob('data/datastore/episodes/episode_data_*.csv')
            if episode_files:
                logger.info(f"Found {len(episode_files)} episode data files to combine")

                # Check if files are valid
                valid_episode_dfs = []
                for f in episode_files:
                    try:
                        df = pd.read_csv(f)
                        if not df.empty:
                            valid_episode_dfs.append(df)
                        else:
                            logger.warning(f"Empty episode data file: {f}")
                    except Exception as e:
                        logger.error(f"Error reading episode file {f}: {str(e)}")

                if valid_episode_dfs:
                    combined_episodes = pd.concat(valid_episode_dfs)
                    combined_episodes.to_csv('data/datastore/episodes/combined_episode_data.csv', index=False)
                    logger.info(f"Combined {len(valid_episode_dfs)} episode data files successfully")

                    # Cleanup after combining
                    for f in episode_files:
                        try:
                            os.remove(f)
                        except Exception as e:
                            logger.error(f"Error removing file {f}: {str(e)}")
                else:
                    logger.warning("No valid episode data files to combine")
                    # Create an empty combined file to avoid errors
                    pd.DataFrame(columns=[
                        'agent_type', 'network', 'episode', 'avg_waiting',
                        'arrived_vehicles', 'total_reward'
                    ]).to_csv('data/datastore/episodes/combined_episode_data.csv', index=False)
            else:
                logger.warning("No episode data files found")
                # Create an empty combined file to avoid errors
                pd.DataFrame(columns=[
                    'agent_type', 'network', 'episode', 'avg_waiting',
                    'arrived_vehicles', 'total_reward'
                ]).to_csv('data/datastore/episodes/combined_episode_data.csv', index=False)

        except Exception as e:
            logger.error(f"Error combining results: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())


class NullDataCollector(Observer):
    """A no-op data collector that doesn't record anything."""

    def on_step_complete(self, data: Dict[str, Any]) -> None:
        pass

    def on_episode_complete(self, data: Dict[str, Any]) -> None:
        pass

    def save_data(self, suffix: str) -> None:
        pass


class MetricsCalculator:
    """Helper class to calculate various metrics for the simulation."""

    def __init__(self, network):
        self.network = network

    def calculate_step_metrics(self, tls_id, action, reward):
        """Calculate metrics for a single step and traffic light."""
        try:
            waiting_time = sum(
                self.network.get_lane_waiting_time(lane)
                for lane in self.network.get_controlled_lanes(tls_id)
            )

            vehicle_count = sum(
                self.network.get_lane_vehicle_count(lane)
                for lane in self.network.get_controlled_lanes(tls_id)
            )

            queue_length = sum(
                self.network.get_lane_queue(lane)
                for lane in self.network.get_controlled_lanes(tls_id)
            )

            phase_duration = self.network.get_phase_duration(tls_id)

            return {
                'tls_id': tls_id,
                'action': action,
                'reward': reward,
                'waiting_time': waiting_time,
                'vehicle_count': vehicle_count,
                'queue_length': queue_length,
                'phase_duration': phase_duration
            }
        except Exception as e:
            logger.error(f"Error calculating metrics for {tls_id}: {str(e)}")
            # Return default metrics to avoid breaking the data collection
            return {
                'tls_id': tls_id,
                'action': action if action is not None else 0,
                'reward': reward if reward is not None else 0,
                'waiting_time': 0.0,
                'vehicle_count': 0,
                'queue_length': 0,
                'phase_duration': 0.0
            }