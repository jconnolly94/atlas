import os
import time
import random
import shutil
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from multiprocessing import Process, Manager, Queue
import traceback
import logging
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from typing import Dict, Any, Optional
from utils.run_manager import RunManager
from queue import Empty  # For data_writer_process error handling

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='runner.log'
)
logger = logging.getLogger('Runner')


# Ensure logger is configured (assuming basicConfig is called elsewhere)
logger = logging.getLogger('DataWriter')

def data_writer_process(queue, step_file_path, episode_file_path):
    """
    Process that reads data dictionaries from a shared queue and writes them to Parquet files.

    Args:
        queue: A multiprocessing Queue for receiving data dictionaries.
        step_file_path: Path where step data will be saved as Parquet.
        episode_file_path: Path where episode data will be saved as Parquet.
    """

    # Define schemas matching the data sent by worker_process
    # Make sure these fields EXACTLY match the keys in the dicts put onto the queue
    step_schema = pa.schema([
        pa.field('agent_type', pa.string()),
        pa.field('network', pa.string()),
        pa.field('episode', pa.int64()),
        pa.field('step', pa.int64()),
        pa.field('tls_id', pa.string()),
        pa.field('action', pa.string()),      # Received as string
        pa.field('reward', pa.float64()),
        pa.field('waiting_time', pa.float64()),
        pa.field('vehicle_count', pa.int64()), # Added based on worker data
        pa.field('queue_length', pa.int64())   # Added based on worker data
        # Add any other fields sent by the worker for step data
    ])

    episode_schema = pa.schema([
        pa.field('agent_type', pa.string()),
        pa.field('network', pa.string()),
        pa.field('episode', pa.int64()),
        pa.field('avg_waiting', pa.float64()),
        pa.field('total_reward', pa.float64()),
        pa.field('total_steps', pa.int64()),      # Added based on worker data
        pa.field('final_throughput', pa.int64()), # Added based on worker data
        pa.field('termination_reason', pa.string()) # Added for early episode termination tracking
        # Add any other fields sent by the worker for episode data
    ])

    # Ensure output directories exist
    try:
        os.makedirs(os.path.dirname(step_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(episode_file_path), exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directories for Parquet files: {e}")
        return # Cannot proceed if directories can't be created

    step_writer = None
    episode_writer = None
    items_processed = 0
    error_count = 0

    try:
        # Initialize Parquet writers
        step_writer = pq.ParquetWriter(step_file_path, step_schema)
        episode_writer = pq.ParquetWriter(episode_file_path, episode_schema)
        logger.info(f"Parquet writers opened for steps: {step_file_path}, episodes: {episode_file_path}")

        # Main processing loop
        while True:
            try:
                # Get next item from queue. Use timeout to prevent potential deadlocks
                # if the sentinel is missed or producers die unexpectedly.
                item = queue.get(timeout=5) # Wait up to 5 seconds

            except Empty:
                # Queue was empty for the timeout duration. Check if producers might still be running.
                # This part requires more sophisticated logic if you want graceful shutdown
                # based on producer status. For now, we just continue waiting.
                logger.debug("Queue empty, continuing to wait...")
                continue
            except (EOFError, BrokenPipeError) as e:
                 logger.error(f"Queue connection error: {e}. Shutting down writer.")
                 break # Exit loop if queue connection breaks
            except Exception as e:
                logger.error(f"Unexpected error getting from queue: {e}. Shutting down writer.")
                break # Exit loop on other queue errors


            # Check for sentinel value to terminate
            if item is None:
                logger.info("Received sentinel (None). Shutting down writer.")
                break

            # Validate item structure
            if not isinstance(item, dict):
                logger.warning(f"Received non-dict item, skipping: {type(item)}")
                error_count += 1
                continue

            item_type = item.pop('type', None) # Get type and remove it from dict
            if item_type not in ['step', 'episode']:
                logger.warning(f"Received item with invalid or missing 'type', skipping: {item_type}")
                error_count += 1
                continue

            # Select appropriate writer and schema
            writer = None
            schema = None
            if item_type == 'step':
                writer = step_writer
                schema = step_schema
            elif item_type == 'episode':
                writer = episode_writer
                schema = episode_schema

            try:
                # Ensure all columns expected by the schema exist in the item, add None if missing.
                # This prevents errors in from_pandas if a worker sends incomplete data.
                record = {}
                for field in schema:
                    record[field.name] = item.get(field.name, None) # Use .get() for safety

                # Convert the single record dictionary to a DataFrame
                df = pd.DataFrame([record])

                # Convert DataFrame to PyArrow Table, explicitly using the schema
                table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)

                # Write the table to the Parquet file
                writer.write_table(table)
                items_processed += 1
                if items_processed % 1000 == 0: # Log progress periodically
                     logger.info(f"Processed {items_processed} items...")

            except pa.ArrowInvalid as arrow_err:
                 logger.error(f"Schema mismatch or data conversion error for {item_type}: {arrow_err}. Item: {record}")
                 error_count += 1
            except Exception as write_err:
                logger.error(f"Error writing {item_type} data to Parquet: {write_err}. Item: {record}")
                error_count += 1
                # Consider adding a small sleep here if errors are persistent to avoid spamming logs
                time.sleep(0.1)

    except Exception as setup_err:
         logger.error(f"Error during data writer setup or main loop: {setup_err}")
         traceback.print_exc() # Print full traceback for setup errors
    finally:
        # Cleanly close writers
        closed_count = 0
        try:
            if step_writer:
                step_writer.close()
                closed_count += 1
                logger.info("Step data Parquet writer closed.")
        except Exception as e:
            logger.error(f"Error closing step writer: {e}")

        try:
            if episode_writer:
                episode_writer.close()
                closed_count += 1
                logger.info("Episode data Parquet writer closed.")
        except Exception as e:
            logger.error(f"Error closing episode writer: {e}")

        logger.info(f"Data writer process finished. Processed items: {items_processed}, Errors: {error_count}, Writers closed: {closed_count}")


class Runner:
    """Main class for running traffic control experiments."""

    def __init__(self):
        """Initialize the runner."""
        self.console = Console()
        self.available_networks = [
            ("Simple4wayCross", "../Networks/Simple4wayCross/simpleCross.sumocfg"),
            ("DublinRd", "../Networks/DublinRd/DublinRd.sumocfg"),
            ("RoxboroArea", "../Networks/RoxboroArea/RoxboroArea.sumocfg"),
        ]
        self.available_agents = [
            "Q-Learning",  # Using lane-level control by default
            "DQN",         # Using lane-level control by default
            "Advanced",
            "Enhanced",
            "Baseline"
        ]
        self.run_manager = RunManager()  # Add this line

    def run_interactive(self):
        """Run the system interactively with user input."""
        self.show_header()

        self.console.print("\n[bold cyan]Experiment Run Management[/bold cyan]")
        choice = input("Start [N]ew run or [R]esume existing run? (N/r): ").strip().lower()

        run_id = None
        config = {}
        start_episode = 0
        num_episodes_to_run = 0
        use_gui = False
        resume_flag = False
        network_config_tuple = None # To store the selected network tuple

        if choice == 'r':
            # --- Resume Logic ---
            self.console.print("\n[green]Available Runs to Resume:[/green]")
            available_runs = self.run_manager.list_runs()
            if not available_runs:
                self.console.print("[yellow]No existing runs found. Starting a new run instead.[/yellow]")
                choice = 'n' # Fallback to new run
            else:
                table = Table(title="Existing Runs")
                table.add_column("#", style="dim")
                table.add_column("Run ID", style="cyan")
                table.add_column("Name", style="green")
                table.add_column("Status", style="yellow")
                table.add_column("Last Episode", style="magenta")
                table.add_column("Started", style="blue")

                valid_indices = []
                for idx, run_meta in enumerate(available_runs):
                    table.add_row(
                        str(idx + 1),
                        run_meta.get('run_id', 'N/A'),
                        run_meta.get('name', 'N/A'),
                        run_meta.get('status', 'N/A'),
                        str(run_meta.get('last_completed_episode', 'N/A')),
                        run_meta.get('start_time', 'N/A')
                    )
                    valid_indices.append(idx + 1)

                self.console.print(table)

                while True:
                    try:
                        resume_choice = int(input(f"\nSelect run to resume (1-{len(available_runs)}): "))
                        if resume_choice in valid_indices:
                            selected_run_meta = available_runs[resume_choice - 1]
                            run_id = selected_run_meta['run_id']
                            config = selected_run_meta.get('config', {})
                            start_episode = selected_run_meta.get('last_completed_episode', 0)
                            total_episodes_in_config = config.get('num_episodes', 0)


                            self.console.print(f"[cyan]Resuming run '{run_id}' from episode {start_episode + 1}.[/cyan]")

                            # Ask if user wants to run *more* episodes or just complete the original count
                            remaining_episodes = total_episodes_in_config - start_episode
                            if remaining_episodes <= 0:
                                self.console.print(f"[yellow]Original target of {total_episodes_in_config} episodes already reached or exceeded.[/yellow]")
                                additional_episodes = int(input("Enter number of *additional* episodes to run (e.g., 10): "))
                                num_episodes_to_run = additional_episodes
                            else:
                                self.console.print(f"[info]{remaining_episodes} episodes remaining to reach original target of {total_episodes_in_config}.[/info]")
                                run_more = input(f"Run additional episodes beyond the original {total_episodes_in_config}? (y/N): ").strip().lower()
                                if run_more == 'y':
                                     additional_episodes = int(input("Enter number of *additional* episodes to run: "))
                                     num_episodes_to_run = remaining_episodes + additional_episodes
                                else:
                                    num_episodes_to_run = remaining_episodes

                            if num_episodes_to_run <= 0:
                                self.console.print("[yellow]No episodes to run. Exiting.[/yellow]")
                                return # Exit if nothing to run

                            # Extract original config details needed for the run
                            network_config_tuple = config.get('network_config')
                            agent_types = config.get('agent_types')
                            use_gui = config.get('use_gui', False) # Get GUI setting from original config
                            resume_flag = True
                            break # Exit selection loop
                        else:
                            self.console.print("[red]Invalid selection.[/red]")
                    except ValueError:
                        self.console.print("[red]Invalid input. Please enter a number.[/red]")
                    except KeyError:
                         self.console.print(f"[red]Error accessing metadata for run '{run_id}'. Starting new run.[/red]")
                         choice = 'n'
                         run_id = None
                         break

        # --- New Run Logic (executed if choice is 'n' initially or after fallback) ---
        if choice != 'r':
            self.console.print("\n[bold green]Configuring New Run[/bold green]")
            run_name = input("Enter a name for this run (optional, press Enter to skip): ").strip() or None
            network_config_tuple = self.select_network()
            agent_types = self.select_agents()
            total_episodes = self.get_num_episodes()
            use_gui = self.get_use_gui()

            config = {
                'network_config': network_config_tuple,
                'agent_types': agent_types,
                'num_episodes': total_episodes,
                'use_gui': use_gui,
                # Add any other relevant config parameters here if needed later
            }
            try:
                run_id = self.run_manager.create_run(config, run_name)
                self.console.print(f"[cyan]Created new run with ID: {run_id}[/cyan]")
                start_episode = 0
                num_episodes_to_run = total_episodes
                resume_flag = False
            except Exception as e:
                self.console.print(f"[bold red]Error creating new run: {e}[/bold red]")
                return # Exit if run creation fails

        # --- Final Check & Run ---
        if run_id is None or not network_config_tuple or not agent_types or num_episodes_to_run <= 0:
             self.console.print("[red]Configuration incomplete. Cannot start experiment.[/red]")
             return

        # Ensure agent_types is available for the run_experiments call
        if 'agent_types' not in config:
             config['agent_types'] = agent_types # Ensure it's set if resuming didn't populate it correctly

        # Prepare final config dictionary to pass (contains original + runtime info)
        run_config_for_workers = {
            'network_config': network_config_tuple,
            'agent_types': agent_types,
            'use_gui': use_gui
            # Don't pass num_episodes here, worker gets num_episodes_to_run directly
        }

        # Call run_experiments with the new signature
        self.run_experiments(
            run_id=run_id,
            config=run_config_for_workers, # Pass the config dict
            start_episode=start_episode,
            num_episodes_to_run=num_episodes_to_run,
            use_gui=use_gui,
            resume_flag=resume_flag
        )

    def show_header(self):
        """Show application header."""
        self.console.print("\n[bold blue]" + "=" * 40)
        self.console.print("[bold blue]Traffic Signal Control System")
        self.console.print("[bold blue]" + "=" * 40)

    def select_network(self):
        """Let user select a network configuration.

        Returns:
            Tuple of (network_name, config_path)
        """
        self.console.print("\n[green]Available Networks:")

        for idx, (name, path) in enumerate(self.available_networks, 1):
            self.console.print(f"  {idx}. {name} ([dim]{path}[/dim])")

        net_choice = int(input("\nSelect network (1-3): ")) - 1
        return self.available_networks[net_choice]

    def select_agents(self):
        """Let user select agent types.

        Returns:
            List of selected agent type names
        """
        self.console.print("\n[green]Available Agents:")

        for idx, name in enumerate(self.available_agents, 1):
            # Add description that all agents use lane-level control
            description = ""
            if name in ["Q-Learning", "DQN"]:
                description = " (using lane-level control)"

            self.console.print(f"  {idx}. {name}{description}")

        agent_choices = input("Select agents (comma-separated, e.g. 1,3,4): ")
        return [self.available_agents[int(c) - 1] for c in agent_choices.split(",")]

    def get_num_episodes(self):
        """Get number of episodes from user.

        Returns:
            Number of episodes to run
        """
        return int(input("\nNumber of training episodes per agent: "))

    def get_use_gui(self):
        """Get GUI preference from user.

        Returns:
            True to use GUI, False for headless
        """
        return input("Use SUMO GUI? (y/n, default: n): ").lower().startswith('y')

    def run_experiments(self, run_id, config, start_episode, num_episodes_to_run, use_gui, resume_flag):
        """Run experiments with the selected configuration.

        Args:
            run_id: The unique identifier for this run
            config: Dictionary containing configuration details
            start_episode: Episode number to start from (0 for new runs)
            num_episodes_to_run: Number of episodes to run per agent
            use_gui: Whether to use the SUMO GUI
            resume_flag: Whether this is resuming an existing run
        """
        # Extract network_config_tuple and agent_types from config dict
        network_config_tuple = config['network_config']
        agent_types = config['agent_types']

        # --- Add Status Update ---
        try:
            self.run_manager.update_run_status(run_id, 'running')
        except Exception as e:
            self.console.print(f"[red]Error updating run status for {run_id}: {e}[/red]")
            # Log warning but continue

        # Show configuration summary
        self.console.print(f"\n[bold green]Starting simulation run: {run_id}[/bold green]")
        self.console.print(f"[green]  Network: {network_config_tuple[0]} ({network_config_tuple[1]})")
        self.console.print(f"[green]  Agents: {', '.join(agent_types)}")
        if resume_flag:
             self.console.print(f"[green]  Mode: Resuming from episode {start_episode + 1}")
             self.console.print(f"[green]  Episodes to run: {num_episodes_to_run}")
        else:
            self.console.print(f"[green]  Mode: New run")
            self.console.print(f"[green]  Episodes: {num_episodes_to_run}") # num_episodes_to_run is total for new runs
        self.console.print(f"[green]  GUI: {'Enabled' if use_gui else 'Disabled'}")

        # Configure port ranges
        base_port = 8873
        port_range_size = 1000

        # Ensure data directories exist
        os.makedirs('data/datastore/steps', exist_ok=True)
        os.makedirs('data/datastore/episodes', exist_ok=True)
        os.makedirs('data/datastore/logs', exist_ok=True)

        # Run experiments in parallel
        with Manager() as manager:
            shared_results = manager.list()
            # Use a simpler progress tracking approach
            progress_flags = manager.dict()
            for agent in agent_types:
                # --- Adjust total_episodes for progress display ---
                total_target_episodes = start_episode + num_episodes_to_run
                progress_flags[agent] = manager.dict({
                    'episode': start_episode, # Start progress from here
                    'total_episodes': total_target_episodes, # Reflect the target episode number
                    'completed': False,
                    'error': False,
                    'status_message': 'Initializing'
                })

            # --- Update Data File Paths ---
            run_path = self.run_manager.get_run_path(run_id)
            data_dir = os.path.join(run_path, 'data')
            os.makedirs(data_dir, exist_ok=True) # Ensure data subdir exists
            step_file_path = os.path.join(data_dir, 'step_data.parquet')
            episode_file_path = os.path.join(data_dir, 'episode_data.parquet')

            # Create shared data queue
            data_queue = manager.Queue()

            # Create and start writer process
            writer_proc = Process(
                target=data_writer_process,  # Assume this function is defined elsewhere
                args=(data_queue, step_file_path, episode_file_path)
            )
            writer_proc.start()

            processes = []

            # Launch worker processes with delay between them
            for i, agent_type in enumerate(agent_types):
                # Each agent gets its own port range
                port_range = (
                    base_port + i * port_range_size,
                    base_port + (i + 1) * port_range_size - 1
                )

                p = Process(
                    target=self.worker_process,
                    args=(
                        run_id,             # NEW position for run_id
                        config,             # Pass the config dict
                        agent_type,
                        port_range,
                        start_episode,      # Start episode number
                        num_episodes_to_run,# Number of episodes to run
                        use_gui,
                        resume_flag,        # NEW
                        shared_results,
                        progress_flags[agent_type],  # Pass individual agent progress dict
                        data_queue          # Pass the shared data queue to the worker
                    )
                )
                processes.append(p)
                p.start()

                self.console.print(f"[blue]Started worker for {agent_type} (PID: {p.pid})")

                # Substantial delay between process starts to avoid port conflicts
                time.sleep(5)

            # Monitor progress with updated signature
            self.monitor_progress(processes, progress_flags, agent_types, start_episode, num_episodes_to_run, run_id)

            # Wait for all processes to complete
            for p in processes:
                p.join()

            # Signal writer shutdown with sentinel value
            data_queue.put(None)

            # Join writer process
            writer_proc.join()

            # --- Add Final Status Update ---
            # Determine overall status based on progress flags or results
            final_status = 'completed'
            for agent in agent_types:
                 if progress_flags[agent].get('error', False):
                     final_status = 'error'
                     break # If one failed, mark the whole run as error
            try:
                # Get the highest completed episode across all agents for metadata
                last_saved_episode = max(progress_flags[agent].get('episode', start_episode) for agent in agent_types)
                self.run_manager.update_run_status(run_id, final_status, last_episode=last_saved_episode)
            except Exception as e:
                 self.console.print(f"[red]Error updating final run status for {run_id}: {e}[/red]")

            # --- Consolidate Worker Logs ---
            self.console.print(f"[dim]Consolidating worker logs for run {run_id}...[/dim]")
            run_log_dir = os.path.join(run_path, 'logs') # run_path is already defined earlier

            # Get worker PIDs from results if available, otherwise reconstruct log filenames
            # It's better if worker_process includes PID in its result dict or we store PIDs
            # For now, let's try reconstructing based on agent_type (less reliable if multiple workers per type)
            # A more robust way: Store p.pid when creating processes.
            worker_pids = [res.get('pid') for res in shared_results if 'pid' in res] # Get PIDs from results

            log_files_found = 0
            log_files_copied = 0
            expected_log_files = []

            # Construct expected log filenames based on PIDs saved in shared_results
            if worker_pids:
                 for res in shared_results:
                     agent_type = res.get('agent')
                     pid = res.get('pid')
                     if agent_type and pid:
                         worker_id = f"{agent_type}_{pid}"
                         expected_log_files.append(f"data/datastore/logs/worker_{worker_id}.log")
            else:
                 # Fallback: Try guessing based only on agent_type (less reliable)
                 logger.warning("Worker PIDs not found in results, attempting log consolidation by agent type (may be inaccurate).")
                 for agent_type in agent_types:
                      # This pattern assumes only one worker per agent type, which is current setup
                      # Need a more robust way if that changes (e.g., storing PIDs)
                      # We might miss logs if PID isn't known
                      pass # Skip fallback for now, requires better PID tracking

            for log_file_path in expected_log_files:
                if os.path.exists(log_file_path):
                    log_files_found += 1
                    try:
                        # Construct destination path
                        log_filename = os.path.basename(log_file_path)
                        dest_path = os.path.join(run_log_dir, log_filename)
                        shutil.move(log_file_path, dest_path) # Move the file
                        log_files_copied += 1
                    except Exception as log_move_err:
                        self.console.print(f"[yellow]Warning: Failed to move log file {log_file_path} to {run_log_dir}: {log_move_err}[/yellow]")
                        logger.warning(f"Failed to move log file {log_file_path} to {run_log_dir}: {log_move_err}")
                else:
                    logger.warning(f"Expected worker log file not found: {log_file_path}")

            self.console.print(f"[dim]Log consolidation: Found {log_files_found}, Copied {log_files_copied} files to {run_log_dir}[/dim]")


            # Convert results
            results = list(shared_results)

        # Display results
        self.display_results(results)

    def monitor_progress(self, processes, progress_flags, agent_types, start_episode, num_episodes_to_run, run_id):
        """Monitor progress with a simplified approach to avoid broken pipe errors.

        Args:
            processes: List of worker processes
            progress_flags: Dictionary with progress flags
            agent_types: List of agent type names
            start_episode: Episode number to start from
            num_episodes_to_run: Number of episodes to run
            run_id: The unique identifier for this run
        """
        # Calculate total work units based on episodes *to be run*
        total_work_units_in_run = len(agent_types) * num_episodes_to_run
        total_target_episodes = start_episode + num_episodes_to_run

        with tqdm(total=100, desc="Overall Progress") as pbar:
            while any(p.is_alive() for p in processes):
                try:
                    # Calculate completed episodes across all agents
                    completed_episodes = 0
                    error_agents = 0
                    completed_agents = 0

                    # Build status text
                    status_parts = []

                    for agent in agent_types:
                        try:
                            agent_progress = progress_flags[agent]

                            current_episode = agent_progress.get('episode', start_episode)
                            episodes_done_this_run = max(0, current_episode - start_episode)
                            status_message = agent_progress.get('status_message', '')

                            if agent_progress.get('error', False):
                                status = f"{agent}: Error - {status_message}"
                                # Consider how much progress to count for errored agents
                                completed_episodes += episodes_done_this_run # Count partial progress
                                error_agents += 1
                            elif agent_progress.get('completed', False):
                                status = f"{agent}: Complete"
                                completed_episodes += num_episodes_to_run # Count all episodes for this agent as done
                                completed_agents += 1
                            else:
                                status = f"{agent}: {current_episode}/{total_target_episodes} - {status_message}"
                                completed_episodes += episodes_done_this_run

                            status_parts.append(status)
                        except:
                            # Handle any dictionary access errors
                            status_parts.append(f"{agent}: Unknown")

                    # Calculate progress percentage based on work units *in this run*
                    if total_work_units_in_run > 0:
                        progress_pct = min(100, int(100 * completed_episodes / total_work_units_in_run))
                    else:
                        progress_pct = 100 if completed_agents == len(agent_types) else 0 # Handle case where num_episodes_to_run is 0

                    # Update progress bar
                    pbar.n = progress_pct

                    # Create status text
                    running = len(agent_types) - completed_agents - error_agents
                    status_text = f"Running: {running} | Complete: {completed_agents} | " + " | ".join(status_parts)

                    # Set progress description
                    pbar.set_description(status_text)
                    pbar.refresh()

                    try:
                        # Determine the safe last completed episode across all *running* agents
                        min_completed_episode_across_running_agents = float('inf')
                        all_agents_still_processing = True # Assume true initially
                        active_agent_count = 0

                        for agent in agent_types:
                            agent_progress = progress_flags.get(agent, {})
                            if not agent_progress: continue # Skip if progress dict not found

                            is_done_or_error = agent_progress.get('completed', False) or agent_progress.get('error', False)
                            current_episode = agent_progress.get('episode', start_episode) # Get reported episode

                            if not is_done_or_error:
                                active_agent_count += 1
                                # We need the last *completed* episode, which is current_episode reported
                                # (assuming progress_dict is updated *after* completion)
                                min_completed_episode_across_running_agents = min(
                                    min_completed_episode_across_running_agents,
                                    current_episode
                                )
                            else:
                                 all_agents_still_processing = False # An agent has finished or failed

                        # Only update metadata if there are still active agents and we have a valid minimum episode
                        if active_agent_count > 0 and min_completed_episode_across_running_agents != float('inf'):
                            # Check if this represents actual progress beyond the start
                            if min_completed_episode_across_running_agents > start_episode:
                                 # Update metadata with the minimum episode completed by all agents still running
                                 self.run_manager.update_run_status(
                                     run_id,
                                     status='running',
                                     last_episode=min_completed_episode_across_running_agents
                                 )

                    except Exception as meta_update_err:
                        # Log error but don't crash the monitor
                        logger.warning(f"Error updating run metadata during monitoring: {meta_update_err}")

                except Exception as e:
                    # Ignore errors in progress monitoring
                    pass

                # Sleep to reduce CPU usage
                time.sleep(1)

            # Ensure progress bar completes when all processes are done
            pbar.n = 100
            pbar.refresh()

    def display_results(self, results):
        """Display experiment results.

        Args:
            results: List of result dictionaries
        """
        table = Table(title="Experiment Results")
        table.add_column("Agent", style="cyan")
        table.add_column("Network", style="green")
        table.add_column("Port", style="blue")
        table.add_column("Episodes", style="magenta")
        table.add_column("Process ID", style="dim")

        for result in results:
            table.add_row(
                result["agent"],
                result["network"],
                str(result["port"]),
                str(result["episodes_completed"]),
                str(result["pid"])
            )

        self.console.print(table)

    @staticmethod
    def worker_process(run_id, config, agent_type, port_range, start_episode, num_episodes_to_run, use_gui, resume_flag, shared_results, progress_dict, data_queue):
        """Worker process for running a single experiment.

        Args:
            run_id: The unique identifier for this run
            config: Dictionary containing configuration details
            agent_type: Agent type to use
            port_range: Range of ports to try
            start_episode: Episode number to start from (0 for new runs)
            num_episodes_to_run: Number of episodes to run from start_episode
            use_gui: Whether to use the SUMO GUI
            resume_flag: Whether this is resuming an existing run
            shared_results: Shared list for collecting results
            progress_dict: Dictionary for tracking this agent's progress
            data_queue: Shared queue for data collection
        """
        # Generate worker ID with PID for uniqueness
        worker_id = f"{agent_type}_{os.getpid()}"
        # --- Define log file path within the standard logs directory ---
        log_dir = "data/datastore/logs"
        os.makedirs(log_dir, exist_ok=True) # Ensure the base log directory exists
        log_file_path = os.path.join(log_dir, f"worker_{worker_id}.log")

        # --- Robust Logging Configuration for this Worker Process ---
        worker_logger = logging.getLogger(worker_id) # Get a logger specific to this worker
        worker_logger.setLevel(logging.INFO) # Set desired level

        # Prevent inheriting handlers from the root logger (important in multiprocessing)
        worker_logger.propagate = False

        # Remove existing handlers (if any were somehow added, e.g., from basicConfig in main process)
        for handler in worker_logger.handlers[:]:
            handler.close()
            worker_logger.removeHandler(handler)

        # Create a file handler specific to this worker's log file
        file_handler = logging.FileHandler(log_file_path, mode='w') # Use 'w' to overwrite each time worker starts
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the handler to this worker's logger
        worker_logger.addHandler(file_handler)

        # Optional: Add a StreamHandler to see worker logs on console during debugging
        # stream_handler = logging.StreamHandler()
        # stream_handler.setFormatter(formatter)
        # worker_logger.addHandler(stream_handler)

        # --- Use worker_logger for all subsequent logging within this method ---
        # Replace all `logger.` calls with `worker_logger.`
        worker_logger.info(f"Worker {worker_id} starting (port range: {port_range}) for run '{run_id}'")
        worker_logger.info(f"Logging configured to file: {log_file_path}")

        # --- Instantiate RunManager ---
        from utils.run_manager import RunManager # Import locally in process
        run_manager = RunManager()

        save_interval = 5 # Save every 5 episodes

        try:

            # Logging setup moved above the try block

            # --- Extract Config ---
            network_config_tuple = config['network_config']
            network_name, config_path = network_config_tuple
            network = None
            port = None

            # Try multiple ports with a large delay between attempts
            max_retries = 5

            for attempt in range(max_retries):
                try:
                    # Close any existing connection
                    try:
                        import traci
                        traci.close()
                    except:
                        pass

                    # Wait a substantial time to ensure ports are free
                    time.sleep(5)

                    # Get random port
                    port = random.randint(port_range[0], port_range[1])
                    worker_logger.info(f"Attempt {attempt + 1}/{max_retries}: Trying port {port}")


                    # Import components here to avoid serialization issues
                    from env.network import Network
                    from env.environment import Environment
                    from utils.data_collector import MetricsCalculator
                    from agents.agent_factory import agent_factory

                    # Create network and start simulation
                    network = Network(config_path, simulation_steps_per_action=5)
                    success = network.start_simulation(use_gui=use_gui, port=port)

                    if not success:
                        worker_logger.error(f"Failed to start SUMO on port {port}")
                        raise Exception("Failed to start SUMO")

                    worker_logger.info(f"Successfully connected to SUMO on port {port}")
                    break
                except Exception as e:
                    worker_logger.error(f"Port {port} connection failed: {str(e)}")
                    traceback.print_exc()
                    time.sleep(5)  # Wait longer before next attempt

            if network is None:
                raise Exception(f"Failed to start SUMO after {max_retries} attempts")

            # Create environment
            env = Environment(network)

            # Create agents for traffic lights
            tls_ids = network.tls_ids
            worker_logger.info(f"Creating/Loading agents for traffic lights: {tls_ids}")

            # --- Agent Instantiation Logic ---
            from agents.agent_factory import agent_factory # Import locally
            agents = {}
            for tls_id in tls_ids:
                agent = agent_factory.create_agent(agent_type, tls_id, network) # Create first
                if resume_flag:
                    agent_state_path = run_manager.get_agent_state_path(run_id, tls_id)
                    worker_logger.info(f"Attempting to load state for agent {tls_id} from {agent_state_path}")
                    try:
                        agent.load_state(agent_state_path) # Load state into existing agent
                    except Exception as load_err:
                         worker_logger.error(f"Failed to load state for agent {tls_id}, starting fresh for this agent: {load_err}")
                         # Continue with the fresh agent instance
                agents[tls_id] = agent

            # --- Load Complete Run Metadata and Extract Early Termination Config ---
            metadata = run_manager.load_metadata(run_id)
            run_config = metadata.get('config', {})
            early_term_config = run_config.get('early_episode_termination', {
                # Provide safe defaults if metadata is missing/old
                'enabled': True,  # Default to disabled if missing
                'max_step_wait_time': 300.0,
                'max_step_queue_length': 40,
                'min_steps_before_check': 100,
                'termination_penalty': -100.0
            })
            worker_logger.info(f"Early termination config: {early_term_config}")

            # Training loop
            worker_logger.info(f"Starting training loop from episode {start_episode + 1} to {start_episode + num_episodes_to_run}")
            episodes_run_this_session = 0

            # --- Adjust Loop Range ---
            for episode in range(start_episode, start_episode + num_episodes_to_run):
                current_episode_number = episode + 1
                try:
                    # Update progress dict - start of episode attempt
                    try:
                        progress_dict['episode'] = current_episode_number
                        progress_dict['status_message'] = f'Running ep {current_episode_number}'
                    except Exception as e:
                        worker_logger.warning(f"Failed to update progress dict: {e}")

                    # Reset environment
                    env.reset()
                    done = False
                    step = 0
                    max_steps = 1000

                    # Episode loop
                    while not done and step < max_steps:
                        # Get actions from agents
                        actions = {}
                        for tls_id, agent in agents.items():
                            state = env.current_states.get(tls_id)
                            if state is not None:
                                action = agent.choose_action(state)
                                actions[tls_id] = action

                        # Perform step in the environment with early termination config
                        next_states, rewards, done = env.step(agents, early_term_config)

                        # Process step data for each agent
                        for tls_id, agent in agents.items():
                            if tls_id in next_states:
                                # Get the effective action for this agent
                                effective_action = actions.get(tls_id)

                                # Calculate metrics for this agent from link states
                                link_states = next_states[tls_id]['link_states']
                                waiting_time = sum(link['waiting_time'] for link in link_states)
                                vehicle_count = sum(link['vehicle_count'] for link in link_states)
                                queue_length = sum(link['queue_length'] for link in link_states)

                                # Create step data dictionary
                                step_data_dict = {
                                    'agent_type': agent_type,
                                    'network': network_name,
                                    'episode': env.episode_number,
                                    'step': env.episode_step - 1,  # Adjusted because env.step already incremented it
                                    'tls_id': tls_id,
                                    'action': str(effective_action) if effective_action is not None else 'None',
                                    'reward': rewards.get(tls_id, 0.0),
                                    'waiting_time': float(waiting_time),
                                    'vehicle_count': int(vehicle_count),
                                    'queue_length': int(queue_length)
                                }

                                # Put on data queue
                                try:
                                    data_queue.put({'type': 'step', **step_data_dict})
                                except Exception as q_err:
                                    worker_logger.error(f"Failed to put step data on queue: {q_err}")

                        step += 1

                    # Handle episode completion
                    if done or step >= max_steps:
                        if not done and step >= max_steps:
                            worker_logger.info(f"Episode {episode + 1} reached max steps without natural termination")

                        # Calculate final episode metrics
                        avg_waiting = np.mean(env.episode_metrics['waiting_times']) if env.episode_metrics['waiting_times'] else 0
                        total_reward = sum(env.episode_metrics['rewards'])
                        final_throughput = network.get_arrived_vehicles_count()

                        # Get termination reason if available, or use a default
                        termination_reason = 'max_steps' if (not done and step >= max_steps) else 'natural'

                        # Create episode data dictionary with termination reason
                        episode_data_dict = {
                            'agent_type': agent_type,
                            'network': network_name,
                            'episode': env.episode_number,
                            'avg_waiting': float(avg_waiting),
                            'total_reward': float(total_reward),
                            'total_steps': env.episode_step,
                            'final_throughput': final_throughput,
                            'termination_reason': termination_reason  # Include termination reason
                        }

                        # Put on data queue
                        try:
                            data_queue.put({'type': 'episode', **episode_data_dict})
                        except Exception as q_err:
                            worker_logger.error(f"Failed to put episode data on queue: {q_err}")

                    worker_logger.info(f"Run '{run_id}', Agent {agent_type}: Episode {current_episode_number} completed at step {step}.")
                    episodes_run_this_session += 1

                    # --- Update progress_dict (end of successful episode) ---
                    try:
                        progress_dict['episode'] = current_episode_number # Ensure it reflects completed episode
                        progress_dict['status_message'] = f'Completed ep {current_episode_number}'
                    except Exception as e:
                        worker_logger.warning(f"Failed to update progress dict: {e}")


                    # --- Periodic Saving ---
                    is_last_episode = (episode == (start_episode + num_episodes_to_run - 1))
                    if (current_episode_number % save_interval == 0) or is_last_episode:
                        worker_logger.info(f"Saving agent states for episode {current_episode_number}")
                        try:
                            for tls_id, agent in agents.items():
                                agent_state_path = run_manager.get_agent_state_path(run_id, tls_id)
                                agent.save_state(agent_state_path)
                            worker_logger.info(f"Agent states saved successfully for episode {current_episode_number}")
                        except Exception as save_err:
                             worker_logger.error(f"Error saving agent states at episode {current_episode_number}: {save_err}")

                except Exception as episode_err:
                    worker_logger.error(f"Run '{run_id}', Agent {agent_type}: Error in episode {current_episode_number}: {episode_err}")
                    traceback.print_exc()
                    try:
                        progress_dict['error'] = True
                        progress_dict['status_message'] = f'Error in ep {current_episode_number}'
                    except Exception as e:
                        worker_logger.warning(f"Failed to update progress dict on error: {e}")
                    break # Stop processing episodes for this worker on error

            # --- Worker Finished ---
            final_episode_completed = start_episode + episodes_run_this_session
            try:
                # Final update to progress dict, ensuring 'completed' is set if no error occurred
                if not progress_dict.get('error', False):
                     progress_dict['completed'] = True
                     progress_dict['episode'] = final_episode_completed # Reflect actual last completed
                     progress_dict['status_message'] = f'Finished {episodes_run_this_session} episodes.'
            except Exception as e:
                worker_logger.warning(f"Failed to update progress dict on completion: {e}")

            # Close connections
            try:
                network.close()
                worker_logger.info("Simulation closed.")
            except Exception as e:
                worker_logger.error(f"Error closing network: {str(e)}")
                traceback.print_exc()

            # Add result to shared list (update episodes completed)
            shared_results.append({
                "agent": agent_type,
                "network": network_name,
                "port": port,
                "episodes_completed": final_episode_completed, # Report total completed episodes
                "episodes_run_this_session": episodes_run_this_session,
                "pid": os.getpid(),
                "run_id": run_id
            })

            worker_logger.info(f"Worker {worker_id} finished. Completed episodes: {final_episode_completed}.")

        except Exception as worker_err:
            worker_logger.error(f"Critical error in worker {worker_id} for run '{run_id}': {worker_err}")
            traceback.print_exc(file=open(log_file_path,'a')) # Append traceback to log file
            try:
                 progress_dict['error'] = True
                 progress_dict['status_message'] = 'Worker process failed'
            except Exception as e:
                worker_logger.warning(f"Failed to update progress dict on critical error: {e}")
            # Ensure cleanup happens if possible
            try: network.close()
            except: pass

            # Try to close any open connections
            try:
                import traci
                traci.close()
            except:
                pass
        finally:
            # --- Important: Close the handler when worker exits ---
            worker_logger.info(f"Worker {worker_id} finalizing.")
            file_handler.close()
            worker_logger.removeHandler(file_handler)


def main():
    """Main entry point."""
    try:
        runner = Runner()
        runner.run_interactive()
    except KeyboardInterrupt:
        console = Console()
        console.print("\n[red]Experiment interrupted by user")
        console.print("[yellow]Cleaning up processes... Please wait")

        # Try to close TRACI connections
        try:
            import traci
            traci.close()
        except:
            pass

    except Exception as e:
        import traceback
        console = Console()
        console.print(f"\n[red]Error in main process: {str(e)}")
        console.print(traceback.format_exc())


if __name__ == "__main__":
    main()