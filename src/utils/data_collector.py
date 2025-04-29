import pandas as pd
import os
import glob
import threading
import time
from typing import Dict, List, Any, Optional, Union, Tuple
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

            # Format action for storage if it's a tuple
            if isinstance(record.get('action'), tuple):
                record['action'] = str(record['action'])

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
                    # Skip creating file if no data exists
                    logger.info(f"Skipping creation of empty step data file: {step_file}")
                else:
                    # Convert to dataframes and save
                    step_df = pd.DataFrame(self.step_data)
                    step_df.to_csv(step_file, index=False)
                    logger.info(f"Step data saved successfully: {step_file}")

                if not self.episode_data:
                    logger.warning(f"No episode data to save for {suffix}")
                    # Skip creating file if no data exists
                    logger.info(f"Skipping creation of empty episode data file: {episode_file}")
                else:
                    # Convert to dataframes and save
                    episode_df = pd.DataFrame(self.episode_data)
                    episode_df.to_csv(episode_file, index=False)
                    logger.info(f"Episode data saved successfully: {episode_file}")

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
                    # Skip creating empty file
                    logger.info("Skipping creation of empty combined step data file")
            else:
                logger.warning("No step data files found")
                # Skip creating empty file
                logger.info("Skipping creation of empty combined step data file")

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
                    # Skip creating empty file
                    logger.info("Skipping creation of empty combined episode data file")
            else:
                logger.warning("No episode data files found")
                # Skip creating empty file
                logger.info("Skipping creation of empty combined episode data file")

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
    """Helper class to calculate metrics for the simulation."""

    def __init__(self, network):
        self.network = network

    def calculate_step_metrics(self, tls_id: str, action: Optional[Union[int, Tuple[int, str]]], reward: float) -> Dict[str, Any]:
        """Calculate metrics for a single step and traffic light.
        
        Supports both traditional phase-based and lane-level control actions.
        
        Args:
            tls_id: ID of the traffic light
            action: Either phase index (int) or link-level action (tuple)
            reward: Reward received for the action
            
        Returns:
            Dictionary of metrics for the step
        """
        try:
            # Calculate standard metrics
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

            metrics = {
                'tls_id': tls_id,
                'reward': reward,
                'waiting_time': waiting_time,
                'vehicle_count': vehicle_count,
                'queue_length': queue_length,
            }
            
            # For lane-level actions, we need special handling
            if isinstance(action, tuple) and len(action) == 2 and isinstance(action[0], int) and isinstance(action[1], str):
                link_index, state = action
                # Store the action as-is - will be converted to string by DataCollector
                metrics['action'] = action
                # For phase duration, use a default value
                metrics['phase_duration'] = 0.0
            elif action is None:
                # Handle None actions (no-ops)
                metrics['action'] = None
                metrics['phase_duration'] = 0.0
            else:
                # For traditional phase actions, use the standard approach
                metrics['action'] = action
                metrics['phase_duration'] = self.network.get_phase_duration(tls_id)
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics for {tls_id}: {str(e)}")
            # Return default metrics to avoid breaking the data collection
            return {
                'tls_id': tls_id,
                'action': str(action) if isinstance(action, tuple) else (action if action is not None else 0),
                'reward': reward if reward is not None else 0,
                'waiting_time': 0.0,
                'vehicle_count': 0,
                'queue_length': 0,
                'phase_duration': 0.0
            }