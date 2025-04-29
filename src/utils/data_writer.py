import multiprocessing
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import os
import time
import logging
from queue import Empty # Use queue.Empty for timeout checks

logger = logging.getLogger('DataWriter')

# Define sentinel value to signal termination
TERMINATION_SENTINEL = None

def data_writer_process(data_queue: multiprocessing.Queue, run_id: str, base_dir: str = 'data/runs'):
    """
    Target function for the process that writes data from the queue to Parquet files.
    """
    run_path = os.path.join(base_dir, run_id)
    data_path = os.path.join(run_path, 'data')
    os.makedirs(data_path, exist_ok=True)
    
    # Create a steps directory to store episode-specific step data
    steps_dir = os.path.join(data_path, 'steps')
    os.makedirs(steps_dir, exist_ok=True)
    
    # Episode data file remains the same
    episode_file = os.path.join(data_path, 'episode_data.parquet')

    # Dictionary to hold writers for each episode
    step_writers = {}
    step_schemas = {}
    episode_writer = None
    episode_schema = None

    buffer_size = 1000  # Write every N records
    # Dictionary to hold step buffers for each episode
    step_buffers = {}
    episode_buffer = []

    logger.info(f"DataWriter started for run {run_id}. Writing to {data_path}")

    try:
        while True:
            try:
                # Get data with a timeout to allow periodic buffer flushing
                record = data_queue.get(timeout=5.0) # Wait up to 5 seconds

                if record is TERMINATION_SENTINEL:
                    logger.info("Termination signal received. Flushing remaining buffers...")
                    break # Exit the loop

                record_type = record.pop('record_type', None) # Get and remove the type marker
                # In runner.py the key is 'type', not 'record_type'
                if record_type is None:
                    record_type = record.pop('type', None)

                if record_type == 'step':
                    # Get the episode number from the record
                    episode_num = record.get('episode', 0)
                    
                    # Initialize buffer for this episode if it doesn't exist
                    if episode_num not in step_buffers:
                        step_buffers[episode_num] = []
                    
                    # Add record to the appropriate episode buffer
                    step_buffers[episode_num].append(record)
                    
                    # Check if buffer needs flushing
                    if len(step_buffers[episode_num]) >= buffer_size:
                        # Construct episode-specific file path
                        step_file = os.path.join(steps_dir, f'step_data_episode_{episode_num}.parquet')
                        step_writers[episode_num], step_schemas[episode_num] = write_batch(
                            step_buffers[episode_num], 
                            step_file, 
                            step_writers.get(episode_num), 
                            step_schemas.get(episode_num)
                        )
                        step_buffers[episode_num].clear()
                        
                elif record_type == 'episode':
                    episode_buffer.append(record)
                    if len(episode_buffer) >= buffer_size:
                        episode_writer, episode_schema = write_batch(episode_buffer, episode_file, episode_writer, episode_schema)
                        episode_buffer.clear()
                else:
                    logger.warning(f"Received record with unknown type: {record_type}")

            except Empty:
                # Timeout occurred, check if buffers need flushing
                for episode_num, buffer in step_buffers.items():
                    if buffer:
                        step_file = os.path.join(steps_dir, f'step_data_episode_{episode_num}.parquet')
                        step_writers[episode_num], step_schemas[episode_num] = write_batch(
                            buffer, 
                            step_file, 
                            step_writers.get(episode_num), 
                            step_schemas.get(episode_num)
                        )
                        buffer.clear()
                        
                if episode_buffer:
                    episode_writer, episode_schema = write_batch(episode_buffer, episode_file, episode_writer, episode_schema)
                    episode_buffer.clear()
                # Continue waiting for data or sentinel
                continue

        # Final flush after receiving sentinel
        for episode_num, buffer in step_buffers.items():
            if buffer:
                step_file = os.path.join(steps_dir, f'step_data_episode_{episode_num}.parquet')
                write_batch(buffer, step_file, step_writers.get(episode_num), step_schemas.get(episode_num))
                
        if episode_buffer:
            write_batch(episode_buffer, episode_file, episode_writer, episode_schema)

    except Exception as e:
        logger.error(f"Error in DataWriter process: {e}", exc_info=True)
    finally:
        # Ensure all writers are closed
        for episode_num, writer in step_writers.items():
            if writer:
                try:
                    writer.close()
                    logger.info(f"Closed step writer for episode {episode_num}")
                except Exception as close_err:
                    logger.error(f"Error closing step writer for episode {episode_num}: {close_err}")
        
        if episode_writer:
            episode_writer.close()
        
        logger.info("DataWriter process finished.")

def write_batch(buffer: list, file_path: str, writer, schema):
    """Helper function to write a batch of records to a Parquet file."""
    if not buffer:
        return writer, schema

    try:
        # Convert list of dicts to Pandas DataFrame first for easier schema handling
        df = pd.DataFrame(buffer)
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)

        if writer is None:
            # Get schema from the first table
            schema = table.schema
            writer = pq.ParquetWriter(file_path, schema)
            logger.info(f"Opened Parquet writer for {os.path.basename(file_path)} with schema: {schema}")

        writer.write_table(table)
        # logger.debug(f"Written batch of {len(buffer)} records to {os.path.basename(file_path)}") # Optional: Debug logging
    except Exception as e:
        logger.error(f"Failed to write batch to {file_path}: {e}", exc_info=True)
        # Decide how to handle failed writes: log, retry, discard? Logging is simplest.

    return writer, schema