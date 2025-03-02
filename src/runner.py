import os
import sys
import time
import random
import pandas as pd
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from multiprocessing import Process, Manager
import traceback
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='runner.log'
)
logger = logging.getLogger('Runner')


class Runner:
    """Main class for running traffic control experiments."""

    def __init__(self):
        """Initialize the runner."""
        self.console = Console()
        self.available_networks = [
            ("Simple4wayCross", "../Networks/Simple4wayCross/simpleCross.sumocfg"),
            ("DublinRd", "../Networks/DublinRd/DublinRd.sumocfg")
        ]
        self.available_agents = [
            "Q-Learning",
            "DQN",
            "Advanced",
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

        net_choice = int(input("\nSelect network (1-2): ")) - 1
        return self.available_networks[net_choice]

    def select_agents(self):
        """Let user select agent types.

        Returns:
            List of selected agent type names
        """
        self.console.print("\n[green]Available Agents:")

        for idx, name in enumerate(self.available_agents, 1):
            self.console.print(f"  {idx}. {name}")

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
            # Add shared progress dictionary for more granular updates
            shared_progress = manager.dict()

            # Initialize progress for each agent
            for agent_type in agent_types:
                shared_progress[agent_type] = {
                    'total_episodes': num_episodes,
                    'current_episode': 0,
                    'current_step': 0,
                    'max_steps': 1000,  # Default, will be updated
                    'status': 'waiting'
                }

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
                        shared_progress  # Pass the shared progress dict
                    )
                )
                processes.append(p)
                p.start()

                self.console.print(f"[blue]Started worker for {agent_type} (PID: {p.pid})")

                # Substantial delay between process starts to avoid port conflicts
                time.sleep(5)

            # Monitor progress
            self.monitor_progress(processes, shared_progress)

            # Wait for all processes to complete
            for p in processes:
                p.join()

            # Convert results
            results = list(shared_results)

        # Display results
        self.display_results(results)

        # Combine data
        if results:
            try:
                self.console.print("[yellow]Combining results from all workers...")

                # Ensure paths exist
                os.makedirs('data/datastore/steps', exist_ok=True)
                os.makedirs('data/datastore/episodes', exist_ok=True)

                # Import here to avoid circular imports
                from utils.data_collector import DataCollector
                DataCollector.combine_results()
                self.console.print("[green]Results combined successfully!")
            except Exception as e:
                self.console.print(f"[red]Error combining results: {str(e)}")
                traceback.print_exc()
        else:
            self.console.print("[yellow]No results to combine.")

    def monitor_progress(self, processes, shared_progress):
        """Monitor progress with detailed updates.

        Args:
            processes: List of worker processes
            shared_progress: Shared dictionary with progress info
        """
        # Calculate total work units
        total_agents = len(processes)

        # Set up progress bar
        with tqdm(total=100, desc="Simulation Progress") as progress:
            while any(p.is_alive() for p in processes):
                # Calculate overall progress percentage
                progress_values = []

                for agent_type, info in shared_progress.items():
                    # Calculate episode progress (0.0 - 1.0)
                    if info['total_episodes'] > 0:
                        episode_progress = info['current_episode'] / info['total_episodes']

                        # Add step progress within current episode
                        if info['current_episode'] < info['total_episodes'] and info['max_steps'] > 0:
                            step_fraction = min(1.0, info['current_step'] / info['max_steps'])
                            episode_fraction = 1.0 / info['total_episodes']
                            agent_progress = episode_progress + (step_fraction * episode_fraction)
                        else:
                            agent_progress = episode_progress
                    else:
                        agent_progress = 0.0

                    progress_values.append(agent_progress)

                # Calculate average progress across all agents
                if progress_values:
                    overall_progress = sum(progress_values) / len(progress_values)
                    progress_percentage = int(overall_progress * 100)

                    # Update progress bar
                    progress.n = min(100, progress_percentage)
                    progress.refresh()

                # Build status text
                status_parts = []
                running_agents = 0

                for agent_type, info in shared_progress.items():
                    if info['status'] == 'completed':
                        status = f"{agent_type}: Done"
                    elif info['status'] == 'error':
                        status = f"{agent_type}: Error"
                    elif info['status'] == 'running':
                        running_agents += 1
                        status = f"{agent_type}: Ep {info['current_episode'] + 1}/{info['total_episodes']} (Step {info['current_step']})"
                    else:
                        status = f"{agent_type}: {info['status'].capitalize()}"

                    status_parts.append(status)

                # Keep status text from getting too long
                if len(status_parts) > 3:
                    short_status = status_parts[:2]
                    short_status.append(f"...and {len(status_parts) - 2} more")
                    status_parts = short_status

                if status_parts:
                    progress.set_description(
                        f"Agents running: {running_agents}/{total_agents} | " + " | ".join(status_parts))

                time.sleep(0.5)  # Update more frequently

            # Ensure progress bar completes
            progress.n = 100
            progress.refresh()

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
    def worker_process(config, agent_type, port_range, num_episodes, use_gui, shared_results, shared_progress):
        """Worker process for running a single experiment.

        Args:
            config: Network configuration
            agent_type: Agent type to use
            port_range: Range of ports to try
            num_episodes: Number of episodes to run
            use_gui: Whether to use the SUMO GUI
            shared_results: Shared list for collecting results
            shared_progress: Shared dictionary for tracking progress
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

            # Update progress to indicate start
            shared_progress[agent_type] = {
                'total_episodes': num_episodes,
                'current_episode': 0,
                'current_step': 0,
                'max_steps': 1000,
                'status': 'starting'
            }

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
                    print(
                        f"Starting SUMO on port {port} with command: sumo -c {config_path} --no-warnings true --no-step-log true --time-to-teleport -1 --seed 42")

                    # Import components here to avoid serialization issues
                    from env.network import Network
                    from env.environment import Environment
                    from utils.data_collector import DataCollector
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

            # Create data collector
            collector = DataCollector(agent_type, network_name)

            # Create environment
            env = Environment(network)
            env.add_observer(collector)

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
                    # Update progress for episode start
                    shared_progress[agent_type] = {
                        'total_episodes': num_episodes,
                        'current_episode': episode,
                        'current_step': 0,
                        'max_steps': 1000,
                        'status': 'running'
                    }

                    # Reset environment
                    env.reset()
                    done = False
                    step = 0
                    max_steps = 1000

                    # Episode loop
                    while not done and step < max_steps:
                        # Get actions from agents
                        next_states, rewards, done = env.step(agents)
                        step += 1

                        # Update progress periodically (every 20 steps)
                        if step % 20 == 0:
                            shared_progress[agent_type]['current_step'] = step
                            shared_progress[agent_type]['max_steps'] = max_steps

                    # Update progress for episode completion
                    shared_progress[agent_type]['current_step'] = step

                    if not done and step >= max_steps:
                        logger.info(f"Episode {episode + 1} reached max steps without natural termination")
                        # Force notify observers about episode completion
                        env.notify_episode_complete({
                            'episode': env.episode_number,
                            'avg_waiting': np.mean(env.episode_metrics['waiting_times']) if env.episode_metrics[
                                'waiting_times'] else 0,
                            'total_reward': sum(env.episode_metrics['rewards']),
                            'arrived_vehicles': network.get_arrived_vehicles_count(),
                            'total_steps': step
                        })

                    logger.info(f"Episode {episode + 1}/{num_episodes} completed at step {step}")
                    episodes_completed = episode + 1

                except Exception as e:
                    logger.error(f"Error in episode {episode + 1}: {str(e)}")
                    traceback.print_exc()
                    shared_progress[agent_type]['status'] = 'error'
                    break

            # Update progress for completion
            shared_progress[agent_type] = {
                'total_episodes': num_episodes,
                'current_episode': num_episodes,
                'current_step': 0,
                'max_steps': 0,
                'status': 'completed'
            }

            # Save data and cleanup
            collector.save_data(f"{network_name}_{agent_type}_{port}")

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

            # Update progress to indicate error
            try:
                shared_progress[agent_type]['status'] = 'error'
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