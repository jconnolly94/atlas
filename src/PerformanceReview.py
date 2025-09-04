# src/utils/summarize_performance.py

import pandas as pd
import os
import warnings
import logging
import re # Import regex for robust parsing

# --- Configuration ---
# Set default Run ID here or leave empty to prompt user
DEFAULT_RUN_ID = "20250331_010632_ContinuanceTest_bddd0326" # Example: Replace with your latest run ID or leave ""
DEFAULT_BASE_DIR = 'data/runs'
# --- End Configuration ---

# Ignore specific pandas warnings if they appear during merge/concat
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


# Configure logging
log_file = 'summarize_run.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'), # Overwrite log each time
        logging.StreamHandler() # Also print logs to console
    ]
)
logger = logging.getLogger('SummarizeRun')

def get_latest_run_id(base_dir: str) -> str | None:
    """Finds the latest run ID based on directory modification time."""
    try:
        run_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not run_dirs:
            return None
        # Sort by modification time, descending
        run_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(base_dir, d)), reverse=True)
        logger.info(f"Latest run found: {run_dirs[0]}")
        return run_dirs[0]
    except FileNotFoundError:
        logger.error(f"Base directory not found: {base_dir}")
        return None
    except Exception as e:
        logger.error(f"Error finding latest run ID: {e}")
        return None

def generalize_termination_reason(reason_str):
    """Collapses detailed termination reasons into general categories."""
    if not isinstance(reason_str, str):
        return 'Unknown' # Handle potential non-string types

    reason_lower = reason_str.lower()

    # Use regex for more robust matching against variations
    if re.search(r'max wait time exceeded', reason_lower):
        return 'MaxWaitTime'
    elif re.search(r'max queue length exceeded', reason_lower):
        return 'MaxQueue'
    elif reason_lower == 'natural':
        return 'Natural'
    elif reason_lower == 'max_steps':
        return 'MaxSteps'
    elif re.search(r'worker process failed', reason_lower) or re.search(r'error in episode', reason_lower):
         return 'WorkerError'
    # Add other specific conditions here if needed
    else:
        # Try to return the original string if it doesn't match known patterns
        # Or define a generic 'Other' category
        # return 'Other'
         return reason_str # Keep unknown reasons as they are for now


def summarize_performance(run_id: str, base_dir: str = 'data/runs'):
    """
    Loads episode data for a run from Parquet and prints a performance summary.

    Args:
        run_id: The unique identifier of the run.
        base_dir: The base directory where runs are stored.
    """
    run_path = os.path.join(base_dir, run_id)
    data_path = os.path.join(run_path, 'data') # Data is now in run_id/data/
    episode_file = os.path.join(data_path, 'episode_data.parquet') # Expect Parquet file

    logger.info(f"\n--- Performance Summary for Run ID: {run_id} ---")

    if not os.path.exists(episode_file):
        logger.error(f"Episode data file not found at: {episode_file}")
        return

    try:
        df = pd.read_parquet(episode_file)
        logger.info(f"Successfully loaded Parquet file: {episode_file}")
    except Exception as e:
        logger.error(f"Failed to read Parquet file: {e}", exc_info=True)
        return

    if df.empty:
        logger.warning("Episode data file is empty. No performance to summarize.")
        return

    logger.info(f"Loaded {len(df)} episode records.")
    required_columns = ['agent_type', 'avg_waiting', 'total_reward', 'final_throughput', 'total_steps', 'termination_reason']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"Missing expected columns: {', '.join(missing_columns)}. Summary might be incomplete.")
        # Proceed with available columns

    # --- Calculate Aggregated Performance Metrics ---
    # Define metrics we want to aggregate
    metrics_to_agg = ['avg_waiting', 'total_reward', 'final_throughput', 'total_steps']
    # Filter list based on columns actually present in the DataFrame
    metrics_present = [m for m in metrics_to_agg if m in df.columns]

    if not metrics_present:
        logger.error("No standard performance metric columns found (avg_waiting, total_reward, etc.). Cannot summarize performance.")
        return

    aggregations = {metric: ['mean', 'std', 'min', 'max'] for metric in metrics_present}
    aggregations['episode'] = ['count'] # Count episodes per agent

    try:
        # Group by agent_type first
        grouped_data = df.groupby('agent_type')
        performance_summary = grouped_data.agg(aggregations)
        # Flatten MultiIndex columns
        performance_summary.columns = ['_'.join(col).strip() for col in performance_summary.columns.values]
        performance_summary.rename(columns={'episode_count': 'episodes_run'}, inplace=True)
        logger.info("Performance metrics aggregated.")
    except KeyError as e:
        logger.error(f"Missing expected column for aggregation: {e}", exc_info=True)
        return
    except Exception as e:
        logger.error(f"Failed during performance aggregation: {e}", exc_info=True)
        return

    # --- Calculate Generalized Termination Reason Counts ---
    if 'termination_reason' in df.columns:
        try:
            # Ensure the column is treated as string, handling potential None/NaN
            df['termination_reason_str'] = df['termination_reason'].astype(str).fillna('Unknown')
            # Apply the generalization function
            df['termination_category'] = df['termination_reason_str'].apply(generalize_termination_reason)

            # Group by agent_type and the new category
            termination_counts = df.groupby(['agent_type', 'termination_category']).size().unstack(fill_value=0)

            # Add prefix to avoid potential column name conflicts
            termination_counts = termination_counts.add_prefix('term_')
            logger.info("Generalized termination reason counts calculated.")
        except Exception as e:
            logger.error(f"Failed during termination reason aggregation: {e}", exc_info=True)
            termination_counts = pd.DataFrame(index=performance_summary.index) # Create empty df to allow merge
    else:
        logger.warning("'termination_reason' column not found. Skipping termination summary.")
        termination_counts = pd.DataFrame(index=performance_summary.index) # Create empty df

    # --- Combine Summaries ---
    try:
        # Use join, ensuring indices align (should align on agent_type)
        full_summary = performance_summary.join(termination_counts, how='left')
        # Fill NaN counts that might result from join with 0
        term_cols = [col for col in full_summary.columns if col.startswith('term_')]
        full_summary[term_cols] = full_summary[term_cols].fillna(0).astype(int)
        # Ensure episodes_run is also integer
        if 'episodes_run' in full_summary.columns:
             full_summary['episodes_run'] = full_summary['episodes_run'].fillna(0).astype(int)

        logger.info("Performance and termination summaries merged.")
    except Exception as e:
        logger.error(f"Failed to merge performance and termination summaries: {e}", exc_info=True)
        full_summary = performance_summary # Fallback to only performance data

    # --- Format and Print ---
    # Start with the episode count
    display_columns = ['episodes_run'] if 'episodes_run' in full_summary.columns else []

    # Add performance metrics dynamically
    for metric in metrics_present:
        mean_col, std_col, min_col, max_col = f'{metric}_mean', f'{metric}_std', f'{metric}_min', f'{metric}_max'
        # Check if all aggregated columns for this metric exist
        if all(col in full_summary.columns for col in [mean_col, std_col, min_col, max_col]):
             display_columns.extend([mean_col, std_col, min_col, max_col])
        else:
             logger.warning(f"Columns for metric '{metric}' not found in aggregated summary. Skipping display.")

    # Add generalized termination columns dynamically and sort them
    term_cols_sorted = sorted([col for col in full_summary.columns if col.startswith('term_')])
    display_columns.extend(term_cols_sorted)

    # Ensure all selected columns actually exist in the final dataframe
    display_columns = [col for col in display_columns if col in full_summary.columns]

    if not display_columns:
        logger.error("No columns available for display in the final summary.")
        return

    print("\nPerformance Metrics (Mean ± Std Dev / Min / Max / Counts):")
    try:
        # Use context manager for display options
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.width', 180, # Adjust width as needed
                               'display.float_format', '{:,.2f}'.format):
            # Select only the columns we want to display
            print(full_summary[display_columns].to_string())
    except Exception as e:
        logger.error(f"Error formatting or printing the summary table: {e}", exc_info=True)
        print("ERROR: Could not display summary table.")

    print("\n--- Summary Complete ---")


if __name__ == "__main__":
    run_id_input = input(f"Enter Run ID to analyze (or press Enter for latest '{DEFAULT_RUN_ID}'): ").strip()
    run_id_to_analyze = run_id_input if run_id_input else DEFAULT_RUN_ID

    if not run_id_to_analyze:
         logger.info("No Run ID provided or found. Trying to find the latest run.")
         run_id_to_analyze = get_latest_run_id(DEFAULT_BASE_DIR)

    if run_id_to_analyze:
        summarize_performance(run_id_to_analyze, DEFAULT_BASE_DIR)
    else:
        logger.error("Could not determine a Run ID to analyze.")