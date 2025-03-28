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

    step_file = os.path.join(data_path, 'step_data.parquet')
    episode_file = os.path.join(data_path, 'episode_data.parquet')

    step_writer = None
    episode_writer = None
    step_schema = None
    episode_schema = None

    buffer_size = 1000  # Write every N records
    step_buffer = []
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

                if record_type == 'step':
                    step_buffer.append(record)
                    if len(step_buffer) >= buffer_size:
                        step_writer, step_schema = write_batch(step_buffer, step_file, step_writer, step_schema)
                        step_buffer.clear()
                elif record_type == 'episode':
                    episode_buffer.append(record)
                    if len(episode_buffer) >= buffer_size:
                        episode_writer, episode_schema = write_batch(episode_buffer, episode_file, episode_writer, episode_schema)
                        episode_buffer.clear()
                else:
                    logger.warning(f"Received record with unknown type: {record_type}")

            except Empty:
                # Timeout occurred, check if buffers need flushing
                if step_buffer:
                    step_writer, step_schema = write_batch(step_buffer, step_file, step_writer, step_schema)
                    step_buffer.clear()
                if episode_buffer:
                    episode_writer, episode_schema = write_batch(episode_buffer, episode_file, episode_writer, episode_schema)
                    episode_buffer.clear()
                # Continue waiting for data or sentinel
                continue

        # Final flush after receiving sentinel
        if step_buffer:
            write_batch(step_buffer, step_file, step_writer, step_schema)
        if episode_buffer:
            write_batch(episode_buffer, episode_file, episode_writer, episode_schema)

    except Exception as e:
        logger.error(f"Error in DataWriter process: {e}", exc_info=True)
    finally:
        # Ensure writers are closed
        if step_writer:
            step_writer.close()
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