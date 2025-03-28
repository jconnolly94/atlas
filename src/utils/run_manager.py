import os
import json
import uuid
from datetime import datetime
import logging
from typing import Dict, List, Optional, Any # Added typing imports

# Configure logging for this module
logger = logging.getLogger('RunManager')
# Basic configuration if run standalone, adjust as needed for project integration
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class RunManager:
    """Manages experiment runs, directories, and metadata."""

    def __init__(self, base_run_dir: str = 'data/runs'):
        """
        Initializes the RunManager.

        Args:
            base_run_dir: The root directory where all run data will be stored.
        """
        self.base_run_dir = base_run_dir
        try:
            os.makedirs(self.base_run_dir, exist_ok=True)
            logger.info(f"RunManager initialized. Base directory: {self.base_run_dir}")
        except OSError as e:
            logger.error(f"Failed to create base run directory {self.base_run_dir}: {e}")
            raise # Stop if base directory cannot be created

    def generate_run_id(self, name: Optional[str] = None) -> str:
        """
        Generates a unique ID for a run, incorporating an optional name.

        Args:
            name: An optional descriptive name for the run.

        Returns:
            A unique string identifier for the run.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_part = str(uuid.uuid4())[:8]
        if name:
            # Basic sanitization for directory names
            safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in name.strip())
            return f"{timestamp}_{safe_name}_{unique_part}"
        else:
            return f"{timestamp}_{unique_part}"

    def create_run(self, config: Dict[str, Any], name: Optional[str] = None) -> str:
        """
        Creates the directory structure and initial metadata file for a new run.

        Args:
            config: A dictionary containing the configuration for this run
                    (e.g., network, agents, episode count).
            name: An optional descriptive name for the run.

        Returns:
            The unique run_id assigned to the created run.

        Raises:
            OSError: If directory creation fails.
        """
        run_id = self.generate_run_id(name)
        run_path = self.get_run_path(run_id)
        agent_dir = os.path.join(run_path, 'agents')
        data_dir = os.path.join(run_path, 'data')
        log_dir = os.path.join(run_path, 'logs') # For potentially copying worker logs

        try:
            os.makedirs(run_path, exist_ok=False) # Error if run_id somehow exists
            os.makedirs(agent_dir)
            os.makedirs(data_dir)
            os.makedirs(log_dir)
            logger.info(f"Created directory structure for run '{run_id}' at {run_path}")

            metadata = {
                'run_id': run_id,
                'name': name if name else 'Unnamed Run',
                'config': config,
                'start_time': datetime.now().isoformat(),
                'status': 'created',
                'last_completed_episode': 0 # Track progress for resuming
            }
            self.save_metadata(run_id, metadata)
            logger.info(f"Initial metadata saved for run '{run_id}'.")
            return run_id
        except FileExistsError:
            logger.error(f"Run directory for '{run_id}' already exists. Potential ID collision.")
            # Handle collision - maybe retry generating ID or raise a specific error
            raise FileExistsError(f"Run ID collision for {run_id}")
        except OSError as e:
            logger.error(f"Failed to create directories for run '{run_id}': {e}")
            # Attempt cleanup if partial creation occurred? Maybe not necessary.
            raise # Re-raise exception

    def get_run_path(self, run_id: str) -> str:
        """
        Constructs the full path to a specific run's directory.

        Args:
            run_id: The unique identifier of the run.

        Returns:
            The absolute or relative path to the run directory.
        """
        return os.path.join(self.base_run_dir, run_id)

    def get_agent_state_path(self, run_id: str, tls_id: str) -> str:
        """
        Constructs the path for storing/loading a specific agent's state within a run.

        Args:
            run_id: The unique identifier of the run.
            tls_id: The identifier of the traffic light (agent).

        Returns:
            The path to the directory where the agent's state should be saved/loaded.
        """
        # Sanitize tls_id for use as a directory name
        safe_tls_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in tls_id)
        return os.path.join(self.get_run_path(run_id), 'agents', safe_tls_id)

    def save_metadata(self, run_id: str, metadata: Dict[str, Any]) -> None:
        """
        Saves the metadata dictionary to the 'metadata.json' file for a given run.

        Args:
            run_id: The unique identifier of the run.
            metadata: The dictionary containing the run's metadata.
        """
        run_path = self.get_run_path(run_id)
        if not os.path.isdir(run_path):
             logger.error(f"Cannot save metadata. Run directory does not exist: {run_path}")
             return # Or raise error

        meta_path = os.path.join(run_path, 'metadata.json')
        try:
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=4, sort_keys=True)
            # logger.debug(f"Metadata saved for run '{run_id}'.") # Reduce log noise
        except TypeError as e:
             logger.error(f"Failed to serialize metadata for run '{run_id}'. Check config contents: {e}")
        except IOError as e:
             logger.error(f"Failed to write metadata file for run '{run_id}': {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred saving metadata for '{run_id}': {e}")


    def load_metadata(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Loads the metadata dictionary from the 'metadata.json' file for a given run.

        Args:
            run_id: The unique identifier of the run.

        Returns:
            The loaded metadata dictionary, or None if the file doesn't exist or an error occurs.
        """
        meta_path = os.path.join(self.get_run_path(run_id), 'metadata.json')
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                # logger.debug(f"Metadata loaded for run '{run_id}'.") # Reduce log noise
                return metadata
        except FileNotFoundError:
            # This is not necessarily an error, could be checking a non-existent run
            # logger.warning(f"Metadata file not found for run '{run_id}' at {meta_path}")
            return None
        except json.JSONDecodeError as e:
             logger.error(f"Failed to decode metadata JSON for run '{run_id}': {e}")
             return None
        except IOError as e:
            logger.error(f"Failed to read metadata file for run '{run_id}': {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred loading metadata for '{run_id}': {e}")
            return None


    def update_run_status(self, run_id: str, status: str, last_episode: Optional[int] = None) -> None:
        """
        Updates the status and optionally the last completed episode in the run's metadata.

        Args:
            run_id: The unique identifier of the run.
            status: The new status string (e.g., 'running', 'completed', 'error').
            last_episode: The episode number that was last successfully completed and saved.
        """
        metadata = self.load_metadata(run_id)
        if metadata:
            metadata['status'] = status
            if last_episode is not None:
                # Only update if the new value is greater than the existing one
                if last_episode > metadata.get('last_completed_episode', -1):
                    metadata['last_completed_episode'] = last_episode
            metadata['last_update_time'] = datetime.now().isoformat()
            self.save_metadata(run_id, metadata)
            logger.info(f"Updated status for run '{run_id}' to '{status}', last episode: {metadata.get('last_completed_episode')}")
        else:
            logger.warning(f"Could not update status for run '{run_id}': Metadata not found.")


    def list_runs(self) -> List[Dict[str, Any]]:
        """
        Scans the base run directory and lists all identified runs by loading their metadata.

        Returns:
            A list of dictionaries, where each dictionary is the metadata for a run.
            Runs without valid metadata might be represented with a minimal dictionary.
        """
        runs = []
        logger.info(f"Scanning for runs in {self.base_run_dir}...")
        try:
            for item in sorted(os.listdir(self.base_run_dir)): # Sort for consistent listing
                run_path = os.path.join(self.base_run_dir, item)
                if os.path.isdir(run_path):
                    metadata = self.load_metadata(item)
                    if metadata:
                        runs.append(metadata)
                    else:
                        # Include directories that look like runs but lack metadata
                        runs.append({
                            'run_id': item,
                            'name': item, # Use directory name as placeholder
                            'status': 'unknown - no metadata',
                            'start_time': None # Indicate unknown start
                            })
            # Sort runs, e.g., by start time descending (handle None start_time)
            runs.sort(key=lambda x: x.get('start_time', '0') or '0', reverse=True)
            logger.info(f"Found {len(runs)} potential runs.")
        except FileNotFoundError:
             logger.error(f"Base run directory not found: {self.base_run_dir}")
        except Exception as e:
            logger.error(f"Failed to list runs: {e}")
        return runs