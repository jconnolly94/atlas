import os
import time
import random
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='runner.log'
)
logger = logging.getLogger('Runner')


def data_writer_process(queue, step_file_path, episode_file_path):
    """
    Process that reads data dictionaries from a shared queue and writes them to Parquet files.
    
    Args:
        queue: A multiprocessing Queue for receiving data dictionaries
        step_file_path: Path where step data will be saved as Parquet
        episode_file_path: Path where episode data will be saved as Parquet
    """
    logger = logging.getLogger('DataWriter')
    
    # Define schemas for step and episode data
    step_schema = pa.schema([
        pa.field('agent_type', pa.string()),
        pa.field('network', pa.string()),
        pa.field('episode', pa.int64()),
        pa.field('step', pa.int64()),
        pa.field('tls_id', pa.string()),
        pa.field('action', pa.string()),  # Action stored as string representation
        pa.field('reward', pa.float64()),
        pa.field('waiting_time', pa.float64()),
    ])
    
    episode_schema = pa.schema([
        pa.field('agent_type', pa.string()),
        pa.field('network', pa.string()),
        pa.field('episode', pa.int64()),
        pa.field('avg_waiting', pa.float64()),
        pa.field('total_reward', pa.float64()),
    ])
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(step_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(episode_file_path), exist_ok=True)
    
    # Initialize Parquet writers
    step_writer = None
    episode_writer = None
    
    try:
        step_writer = pq.ParquetWriter(step_file_path, step_schema)
        episode_writer = pq.ParquetWriter(episode_file_path, episode_schema)
        logger.info(f"Parquet writers opened for {step_file_path} and {episode_file_path}")
        
        # Main processing loop
        while True:
            try:
                # Get next item from queue
                item = queue.get()
                
                # Check for sentinel value
                if item is None:
                    logger.info("Received sentinel. Shutting down writer.")
                    break
                
                # Validate item is a dictionary
                if not isinstance(item, dict):
                    logger.warning(f"Received non-dict item: {item}")
                    continue
                
                # Determine item type
                item_type = item.pop('type', None)
                
                if item_type is None:
                    logger.warning(f"Item missing 'type' field: {item}")
                    continue
                
                # Process based on item type
                try:
                    if item_type == 'step':
                        # Convert dict to DataFrame
                        df = pd.DataFrame([item])
                        
                        # Convert DataFrame to PyArrow Table with schema
                        table = pa.Table.from_pandas(df, schema=step_schema)
                        
                        # Write to Parquet file
                        step_writer.write_table(table)
                        
                    elif item_type == 'episode':
                        # Convert dict to DataFrame
                        df = pd.DataFrame([item])
                        
                        # Convert DataFrame to PyArrow Table with schema
                        table = pa.Table.from_pandas(df, schema=episode_schema)
                        
                        # Write to Parquet file
                        episode_writer.write_table(table)
                        
                    else:
                        logger.warning(f"Unknown item type: {item_type}")
                        
                except Exception as e:
                    logger.error(f"Error processing {item_type} data: {e}")
                    
            except Exception as e:
                logger.error(f"Error in data writer process: {e}")
                
    finally:
        # Clean up resources
        if step_writer:
            step_writer.close()
            logger.info("Step data Parquet writer closed.")
            
        if episode_writer:
            episode_writer.close()
            logger.info("Episode data Parquet writer closed.")


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

    def run_interactive(self):
        """Run the system interactively with user input."""
        self.show_header()

        # Get user selections
        network_config = self.select_network()
        agent_types = self.select_agents()
        num_episodes = self.get_num_episodes()
        use_gui = self.get_use_gui()

        # Run experiments
        self.run_experiments(network_config, agent_types, num_episodes, use_gui)

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

    def run_experiments(self, network_config, agent_types, num_episodes, use_gui):
        """Run experiments with the selected configuration.

        Args:
            network_config: Tuple of (network_name, config_path)
            agent_types: List of agent type names
            num_episodes: Number of episodes to run per agent
            use_gui: Whether to use the SUMO GUI
        """
        # Show configuration summary
        self.console.print(f"\n[bold green]Starting simulation with:")
        self.console.print(f"[green]  Network: {network_config[0]} ({network_config[1]})")
        self.console.print(f"[green]  Agents: {', '.join(agent_types)}")
        self.console.print(f"[green]  Episodes: {num_episodes} per agent")
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
                progress_flags[agent] = manager.dict({
                    'episode': 0,
                    'total_episodes': num_episodes,
                    'completed': False,
                    'error': False
                })
            
            # Create shared data queue
            data_queue = manager.Queue()
            
            # Placeholder paths - will be dynamic later
            step_file_path = 'temp_step_data.parquet'
            episode_file_path = 'temp_episode_data.parquet'
            
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
                        network_config,
                        agent_type,
                        port_range,
                        num_episodes,
                        use_gui,
                        shared_results,
                        progress_flags[agent_type],  # Pass individual agent progress dict
                        data_queue  # Pass the shared data queue to the worker
                    )
                )
                processes.append(p)
                p.start()

                self.console.print(f"[blue]Started worker for {agent_type} (PID: {p.pid})")

                # Substantial delay between process starts to avoid port conflicts
                time.sleep(5)

            # Monitor progress
            self.monitor_progress(processes, progress_flags, agent_types, num_episodes)

            # Wait for all processes to complete
            for p in processes:
                p.join()
                
            # Signal writer shutdown with sentinel value
            data_queue.put(None)
            
            # Join writer process
            writer_proc.join()

            # Convert results
            results = list(shared_results)

        # Display results
        self.display_results(results)

    def monitor_progress(self, processes, progress_flags, agent_types, total_episodes):
        """Monitor progress with a simplified approach to avoid broken pipe errors.

        Args:
            processes: List of worker processes
            progress_flags: Dictionary with progress flags
            agent_types: List of agent type names
            total_episodes: Total episodes per agent
        """
        # Calculate total work units (number of episodes across all agents)
        total_work = len(agent_types) * total_episodes

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

                            if agent_progress.get('error', False):
                                status = f"{agent}: Error"
                                error_agents += 1
                            elif agent_progress.get('completed', False):
                                status = f"{agent}: Complete"
                                completed_episodes += total_episodes
                                completed_agents += 1
                            else:
                                current_episode = agent_progress.get('episode', 0)
                                status = f"{agent}: {current_episode}/{total_episodes}"
                                completed_episodes += current_episode

                            status_parts.append(status)
                        except:
                            # Handle any dictionary access errors
                            status_parts.append(f"{agent}: Unknown")

                    # Calculate progress percentage
                    if total_work > 0:
                        progress_pct = min(100, int(100 * completed_episodes / total_work))
                    else:
                        progress_pct = 0

                    # Update progress bar
                    pbar.n = progress_pct

                    # Create status text
                    running = len(agent_types) - completed_agents - error_agents
                    status_text = f"Running: {running} | Complete: {completed_agents} | " + " | ".join(status_parts)

                    # Set progress description
                    pbar.set_description(status_text)
                    pbar.refresh()

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
    def worker_process(config, agent_type, port_range, num_episodes, use_gui, shared_results, progress_dict, data_queue):
        """Worker process for running a single experiment.

        Args:
            config: Network configuration
            agent_type: Agent type to use
            port_range: Range of ports to try
            num_episodes: Number of episodes to run
            use_gui: Whether to use the SUMO GUI
            shared_results: Shared list for collecting results
            progress_dict: Dictionary for tracking this agent's progress
            data_queue: Shared queue for data collection
        """
        try:
            # Generate worker ID
            worker_id = f"{agent_type}"
            log_file = f"data/datastore/logs/worker_{worker_id}.log"

            # Configure logging
            import logging
            logging.basicConfig(
                filename=log_file,
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            logger = logging.getLogger(worker_id)
            logger.info(f"Worker {worker_id} starting (port range: {port_range})")

            # Create network with retries
            network_name, config_path = config
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
                    logger.info(f"Attempt {attempt + 1}/{max_retries}: Trying port {port}")


                    # Import components here to avoid serialization issues
                    from env.network import Network
                    from env.environment import Environment
                    from utils.data_collector import MetricsCalculator
                    from agents.agent_factory import agent_factory

                    # Create network and start simulation
                    network = Network(config_path, simulation_steps_per_action=5)
                    success = network.start_simulation(use_gui=use_gui, port=port)

                    if not success:
                        logger.error(f"Failed to start SUMO on port {port}")
                        raise Exception("Failed to start SUMO")

                    logger.info(f"Successfully connected to SUMO on port {port}")
                    break
                except Exception as e:
                    logger.error(f"Port {port} connection failed: {str(e)}")
                    traceback.print_exc()
                    time.sleep(5)  # Wait longer before next attempt

            if network is None:
                raise Exception(f"Failed to start SUMO after {max_retries} attempts")

            # Create environment
            env = Environment(network)

            # Create agents for traffic lights
            tls_ids = network.tls_ids
            logger.info(f"Creating agents for traffic lights: {tls_ids}")

            # Create agents using factory
            agents = {}

            for tls_id in tls_ids:
                agents[tls_id] = agent_factory.create_agent(agent_type, tls_id, network)

            # Training loop
            logger.info(f"Starting {num_episodes} episodes for {agent_type}")

            episodes_completed = 0
            for episode in range(num_episodes):
                try:
                    # Update progress safely (only once per episode)
                    try:
                        progress_dict['episode'] = episode
                    except:
                        # Ignore errors updating progress
                        pass

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

                        # Perform step in the environment
                        next_states, rewards, done = env.step(agents)
                        
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
                                    logger.error(f"Failed to put step data on queue: {q_err}")
                        
                        step += 1

                    # Handle episode completion
                    if done or step >= max_steps:
                        if not done and step >= max_steps:
                            logger.info(f"Episode {episode + 1} reached max steps without natural termination")
                        
                        # Calculate final episode metrics
                        avg_waiting = np.mean(env.episode_metrics['waiting_times']) if env.episode_metrics['waiting_times'] else 0
                        total_reward = sum(env.episode_metrics['rewards'])
                        final_throughput = network.get_arrived_vehicles_count()
                        
                        # Create episode data dictionary
                        episode_data_dict = {
                            'agent_type': agent_type,
                            'network': network_name,
                            'episode': env.episode_number,
                            'avg_waiting': float(avg_waiting),
                            'total_reward': float(total_reward),
                            'total_steps': env.episode_step,
                            'final_throughput': final_throughput
                        }
                        
                        # Put on data queue
                        try:
                            data_queue.put({'type': 'episode', **episode_data_dict})
                        except Exception as q_err:
                            logger.error(f"Failed to put episode data on queue: {q_err}")

                    logger.info(f"Episode {episode + 1}/{num_episodes} completed at step {step}")
                    episodes_completed = episode + 1

                except Exception as e:
                    logger.error(f"Error in episode {episode + 1}: {str(e)}")
                    traceback.print_exc()
                    # Mark progress as error
                    try:
                        progress_dict['error'] = True
                    except:
                        pass
                    break

            # Mark progress as completed
            try:
                progress_dict['completed'] = True
                progress_dict['episode'] = num_episodes
            except:
                pass

            # Close connections
            try:
                network.close()
                logger.info("Simulation closed.")
            except Exception as e:
                logger.error(f"Error closing network: {str(e)}")
                traceback.print_exc()

            # Add result to shared list
            shared_results.append({
                "agent": agent_type,
                "network": network_name,
                "port": port,
                "episodes_completed": episodes_completed,
                "pid": os.getpid()
            })

            logger.info(f"Worker {worker_id} completed successfully")

        except Exception as e:
            print(f"Error in worker for {agent_type} (PID {os.getpid()}): {str(e)}")
            traceback.print_exc()

            # Try to mark progress as error
            try:
                progress_dict['error'] = True
            except:
                pass

            # Try to close any open connections
            try:
                import traci
                traci.close()
            except:
                pass


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