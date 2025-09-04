import os
import logging
import time
import random
import traceback

def setup_worker_logging(worker_id):
    """Configure logging for a worker process."""
    log_dir = "data/datastore/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"worker_{worker_id}.log")

    worker_logger = logging.getLogger(worker_id)
    worker_logger.setLevel(logging.INFO)
    worker_logger.propagate = False

    for handler in worker_logger.handlers[:]:
        handler.close()
        worker_logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file_path, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    worker_logger.addHandler(file_handler)

    return worker_logger, file_handler

def start_sumo_network(config_path, port_range, use_gui, worker_logger):
    """Start the SUMO simulation and return the network object and port."""
    from env.network import Network
    import traci

    max_retries = 5
    network = None
    port = None

    for attempt in range(max_retries):
        try:
            try:
                traci.close()
            except:
                pass

            time.sleep(5)

            port = random.randint(port_range[0], port_range[1])
            worker_logger.info(f"Attempt {attempt + 1}/{max_retries}: Trying port {port}")

            network = Network(config_path, simulation_steps_per_action=5)
            success = network.start_simulation(use_gui=use_gui, port=port)

            if not success:
                worker_logger.error(f"Failed to start SUMO on port {port}")
                raise Exception("Failed to start SUMO")

            worker_logger.info(f"Successfully connected to SUMO on port {port}")
            return network, port

        except Exception as e:
            worker_logger.error(f"Port {port} connection failed: {str(e)}")
            traceback.print_exc()
            time.sleep(5)

    raise Exception(f"Failed to start SUMO after {max_retries} attempts")