# src/compare_run_performance.py

import pandas as pd
import os
import matplotlib.pyplot as plt
import logging
import argparse
import warnings
from typing import List, Optional

# --- Configuration ---
DEFAULT_BASE_DIR = 'data/runs'
DEFAULT_ROLLING_WINDOW = 20 # Number of episodes for smoothing average
# --- End Configuration ---

# Ignore specific pandas warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Configure logging
log_file = 'compare_run.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'), # Overwrite log each time
        logging.StreamHandler() # Also print logs to console
    ]
)
logger = logging.getLogger('CompareRunPerformance')

# --- Helper Function (from PerformanceReview.py) ---
def get_latest_run_id(base_dir: str) -> Optional[str]:
    """Finds the latest run ID based on directory modification time."""
    try:
        run_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not run_dirs:
            return None
        run_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(base_dir, d)), reverse=True)
        logger.info(f"Latest run found: {run_dirs[0]}")
        return run_dirs[0]
    except FileNotFoundError:
        logger.error(f"Base directory not found: {base_dir}")
        return None
    except Exception as e:
        logger.error(f"Error finding latest run ID: {e}")
        return None

# --- Main Comparison Function ---
def compare_performance_over_time(run_id: str, base_dir: str = DEFAULT_BASE_DIR, window_size: int = DEFAULT_ROLLING_WINDOW):
    """
    Loads episode data for a run and generates plots comparing agent performance over time.

    Args:
        run_id: The unique identifier of the run.
        base_dir: The base directory where runs are stored.
        window_size: The number of episodes for the rolling average window.
    """
    run_path = os.path.join(base_dir, run_id)
    data_path = os.path.join(run_path, 'data')
    plot_path = os.path.join(run_path, 'plots') # Directory to save plots
    episode_file = os.path.join(data_path, 'episode_data.parquet')

    logger.info(f"\n--- Performance Comparison Over Time for Run ID: {run_id} ---")
    logger.info(f"Using rolling window size: {window_size}")

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
        logger.warning("Episode data file is empty. No performance to compare.")
        return

    # Ensure 'episode' column exists and is numeric for sorting
    if 'episode' not in df.columns:
        logger.error("Missing 'episode' column in data. Cannot plot over time.")
        return
    try:
        # Convert episode to numeric, coercing errors (though it should be int64)
        df['episode'] = pd.to_numeric(df['episode'], errors='coerce')
        df.dropna(subset=['episode'], inplace=True) # Remove rows where episode couldn't be parsed
        df['episode'] = df['episode'].astype(int)
    except Exception as e:
        logger.error(f"Error processing 'episode' column: {e}. Cannot plot over time.")
        return


    # Ensure 'agent_type' column exists
    if 'agent_type' not in df.columns:
        logger.error("Missing 'agent_type' column in data. Cannot compare agents.")
        return

    # Create plot directory if it doesn't exist
    try:
        os.makedirs(plot_path, exist_ok=True)
        logger.info(f"Plots will be saved to: {plot_path}")
    except OSError as e:
        logger.error(f"Failed to create plot directory: {e}. Plots cannot be saved.")
        return # Cannot proceed without a place to save plots

    # --- Identify Agents and Metrics ---
    agent_types = df['agent_type'].unique()
    logger.info(f"Agents found in data: {', '.join(agent_types)}")

    # Define metrics to plot (ensure these columns exist in episode_data.parquet)
    metrics_to_plot = ['avg_waiting', 'total_reward', 'final_throughput', 'total_steps']
    available_metrics = [m for m in metrics_to_plot if m in df.columns]

    if not available_metrics:
        logger.error("No standard performance metric columns found in data. Cannot generate plots.")
        return

    logger.info(f"Plotting metrics: {', '.join(available_metrics)}")

    # --- Generate Plots for Each Metric ---
    for metric in available_metrics:
        plt.figure(figsize=(12, 7)) # Create a new figure for each metric

        for agent in agent_types:
            # Filter data for the current agent and sort by episode
            agent_df = df[df['agent_type'] == agent].sort_values('episode').reset_index()

            if agent_df.empty:
                logger.warning(f"No data found for agent '{agent}' for metric '{metric}'. Skipping.")
                continue

            # Calculate rolling mean and standard deviation
            # min_periods=1 ensures we get output even if window isn't full at the start
            rolling_mean = agent_df[metric].rolling(window=window_size, min_periods=1).mean()
            rolling_std = agent_df[metric].rolling(window=window_size, min_periods=1).std()

            # Plot the smoothed line (rolling mean)
            plt.plot(agent_df['episode'], rolling_mean, label=f'{agent}')

            # Plot the shaded error band (±1 standard deviation)
            plt.fill_between(agent_df['episode'],
                             rolling_mean - rolling_std,
                             rolling_mean + rolling_std,
                             alpha=0.2) # Use transparency for the fill

        # --- Finalize Plot ---
        plt.title(f'Comparison of {metric.replace("_", " ").title()} (Smoothed, Window={window_size})')
        plt.xlabel('Episode')
        plt.ylabel(f'Smoothed {metric.replace("_", " ").title()}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout() # Adjust layout to prevent labels overlapping

        # Save the plot
        plot_filename = os.path.join(plot_path, f'{run_id}_{metric}_comparison.png')
        try:
            plt.savefig(plot_filename)
            logger.info(f"Saved plot: {plot_filename}")
        except Exception as e:
            logger.error(f"Failed to save plot {plot_filename}: {e}")

        plt.close() # Close the figure to free memory

    logger.info("\n--- Comparison Complete ---")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare agent performance over time for a specific run.")
    parser.add_argument(
        "--run_id",
        type=str,
        default=None, # Default to None, will use latest or prompt if not provided
        help="The unique identifier of the run to analyze. If omitted, uses the latest run."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=DEFAULT_BASE_DIR,
        help=f"The base directory where runs are stored (default: {DEFAULT_BASE_DIR})."
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_ROLLING_WINDOW,
        help=f"The rolling window size for smoothing metrics (default: {DEFAULT_ROLLING_WINDOW})."
    )

    args = parser.parse_args()

    run_id_to_analyze = args.run_id

    if not run_id_to_analyze:
         logger.info("No Run ID provided. Trying to find the latest run.")
         run_id_to_analyze = get_latest_run_id(args.base_dir)

    if run_id_to_analyze:
        compare_performance_over_time(run_id_to_analyze, args.base_dir, args.window)
    else:
        logger.error("Could not determine a Run ID to analyze. Please specify one using --run_id.")