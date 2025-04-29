import numpy as np
import traci
import networkx as nx
import os
import warnings
import logging
from typing import List, Dict, Any, Tuple, Optional, Set

from src.utils.network_exporter import NetworkExporter


class NetworkInterface:
    """Interface for interacting with the traffic network."""

    # Phase-based control methods (current approach)
    def get_current_phase_index(self, tls_id: str) -> int:
        """Get the current phase index for a traffic light."""
        pass

    def get_traffic_light_phases(self, tls_id: str) -> List[Any]:
        """Get all phases defined for a traffic light."""
        pass

    def set_traffic_light_phase(self, tls_id: str, phase: int) -> None:
        """Set phase for a traffic light."""
        pass

    # Lane metrics (still needed for state representation)
    def get_controlled_lanes(self, tls_id: str) -> List[str]:
        """Get lanes controlled by a traffic light."""
        pass

    def get_lane_waiting_time(self, lane_id: str) -> float:
        """Get waiting time for a lane."""
        pass

    def get_lane_vehicle_count(self, lane_id: str) -> int:
        """Get vehicle count for a lane."""
        pass

    def get_lane_queue(self, lane_id: str) -> int:
        """Get queue length for a lane."""
        pass

    # Additional methods needed for state representation
    def get_red_yellow_green_state(self, tls_id: str) -> str:
        """Get the current signal state string (still used for state representation)."""
        pass

    def get_link_metrics(self, tls_id: str) -> List[Dict[str, Any]]:
        """Get performance metrics for each link controlled by a traffic light."""
        pass


class Network(NetworkInterface):
    """Implementation of the network interface using SUMO and TraCI."""

    def __init__(self, sumo_config_file: str, simulation_steps_per_action: int = 5):
        """Initialize the network with phase-based control configuration.

        Args:
            sumo_config_file: Path to SUMO configuration file
            simulation_steps_per_action: Number of simulation steps per action
        """
        self.sumo_config_file = sumo_config_file
        self.simulation_steps_per_action = simulation_steps_per_action
        self.graph = nx.DiGraph()  # For visualization only
        self.arrived_so_far = 0
        self.tls_ids = []
        self.port = None

        # Extract network name from config file path
        self.network_name = os.path.splitext(os.path.basename(sumo_config_file))[0]

        # Flag to track if network data has been exported
        self.network_exported = False

        # Initialize logger
        self.logger = logging.getLogger(f"Network_{id(self)}")  # Or just "Network"
        self.logger.setLevel(logging.INFO)  # Or desired level
        # Add handlers if needed, or rely on root logger config
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    # REMOVED THE FOLLOWING LANE-LEVEL CONTROL METHODS:
    # - set_red_yellow_green_state
    # - change_specific_link
    # - check_physical_conflicts
    # - _simple_conflict_check

    # Lane metrics implementation - RETAINED for state representation

    def get_controlled_lanes(self, tls_id: str) -> List[str]:
        """Get lanes controlled by a traffic light."""
        return traci.trafficlight.getControlledLanes(tls_id)

    def get_lane_waiting_time(self, lane_id: str) -> float:
        """Get waiting time for a lane."""
        return traci.lane.getWaitingTime(lane_id)

    def get_lane_vehicle_count(self, lane_id: str) -> int:
        """Get vehicle count for a lane."""
        return traci.lane.getLastStepVehicleNumber(lane_id)

    def get_lane_queue(self, lane_id: str) -> int:
        """Get queue length for a lane."""
        return traci.lane.getLastStepHaltingNumber(lane_id)

    # Required methods for state representation
    def get_controlled_links(self, tls_id: str) -> List[List[Tuple[str, str, str]]]:
        """Get links controlled by a traffic light with mapping to lanes.

        Args:
            tls_id: ID of the traffic light

        Returns:
            List of links, where each link is a tuple of (fromLane, toLane, internalLane)
        """
        try:
            return traci.trafficlight.getControlledLinks(tls_id)
        except traci.exceptions.TraCIException as e:
            self.logger.warning(f"Could not get controlled links for {tls_id}: {e}")
            return []

    def get_red_yellow_green_state(self, tls_id: str) -> str:
        """Get the current signal state string.

        Args:
            tls_id: ID of the traffic light

        Returns:
            Current signal state string (e.g., "GrGr")
        """
        try:
            return traci.trafficlight.getRedYellowGreenState(tls_id)
        except traci.exceptions.TraCIException as e:
            self.logger.warning(f"Could not get RYG state for {tls_id}: {e}")
            return ""

    def get_link_metrics(self, tls_id: str) -> List[Dict[str, Any]]:
        """Get performance metrics for each link controlled by a traffic light.

        Args:
            tls_id: ID of the traffic light

        Returns:
            List of dictionaries with metrics for each link
        """
        links = self.get_controlled_links(tls_id)
        if not links:  # Empty list indicates error
            self.logger.warning(f"Cannot get link metrics - no links for {tls_id}")
            return []

        link_metrics = []

        for i, link_group in enumerate(links):
            if not link_group:  # Skip empty links
                continue

            try:
                link = link_group[0]  # Take the first link in the group
                from_lane = link[0]

                link_metrics.append({
                    'index': i,
                    'from_lane': from_lane,
                    'waiting_time': self.get_lane_waiting_time(from_lane),
                    'vehicle_count': self.get_lane_vehicle_count(from_lane),
                    'queue_length': self.get_lane_queue(from_lane)
                })
            except Exception as e:
                self.logger.warning(f"Error getting metrics for link {i} of {tls_id}: {e}")

        return link_metrics

    # Core phase-based control methods
    def get_current_phase_index(self, tls_id: str) -> int:
        """Get the current phase index for a traffic light.

        Args:
            tls_id: ID of the traffic light

        Returns:
            Current phase index
        """
        try:
            return traci.trafficlight.getPhase(tls_id)
        except Exception as e:
            self.logger.error(f"Error getting current phase for {tls_id}: {e}")
            return 0  # Default fallback value

    def get_traffic_light_phases(self, tls_id: str) -> List[Any]:
        """Get all phases defined for a traffic light.

        Args:
            tls_id: ID of the traffic light

        Returns:
            List of phase objects
        """
        try:
            logics = traci.trafficlight.getAllProgramLogics(tls_id)
            if not logics:
                raise ValueError(f"No traffic light programs found for TLS {tls_id}")
            return logics[0].phases
        except Exception as e:
            self.logger.error(f"Error getting traffic light phases for {tls_id}: {e}")
            return []  # Empty list as fallback

    def set_traffic_light_phase(self, tls_id: str, phase: int) -> bool:
        """Set phase for a traffic light.

        Args:
            tls_id: ID of the traffic light
            phase: Phase index

        Returns:
            True if successful, False otherwise
        """
        try:
            traci.trafficlight.setPhase(tls_id, phase)
            return True
        except Exception as e:
            self.logger.error(f"Network: Could not set phase {phase} for TLS {tls_id}: {e}")
            return False

    # Simulation management methods
    def build_from_sumo(self) -> None:
        """Build the graph from SUMO's current network (for visualization)."""
        self.graph.clear()

        # Add junctions as nodes
        for junction_id in traci.junction.getIDList():
            self.graph.add_node(junction_id, type='junction')

        # Add edges
        for edge_id in traci.edge.getIDList():
            num_lanes = traci.edge.getLaneNumber(edge_id)
            from_node = traci.edge.getFromJunction(edge_id)
            to_node = traci.edge.getToJunction(edge_id)

            # Add only lanes that actually exist in the SUMO network
            for lane_index in range(num_lanes):
                lane_id = f"{edge_id}_{lane_index}"
                self.graph.add_edge(from_node, to_node, id=lane_id, waiting_time=0)

        # Export network data for visualization if not already done
        if not self.network_exported:
            try:
                export_result = NetworkExporter.export_network_data(self, self.network_name)
                self.network_exported = True
                self.logger.info(f"Exported network visualization data for {self.network_name}")
            except Exception as e:
                self.logger.warning(f"Network export failed: {e}")

    def start_simulation(self, use_gui: bool = False, port: Optional[int] = None) -> bool:
        """Start the SUMO simulation.

        Args:
            use_gui: Whether to use the SUMO GUI
            port: Port for TraCI connection

        Returns:
            True if successful, False otherwise
        """
        self.port = port
        sumo_binary = "sumo-gui" if use_gui else "sumo"

        # Log file
        import os
        os.makedirs("data/datastore/logs", exist_ok=True)
        logfile = open(f"data/datastore/logs/sumo.log", "w")

        # Command
        cmd = [
            sumo_binary,
            "-c", self.sumo_config_file,
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--time-to-teleport", "-1",
            "--seed", "42"  # Fixed seed for reproducibility
        ]

        try:
            self.logger.info(f"Starting SUMO on port {port} with command: {' '.join(cmd)}")
            traci.start(cmd, port=port, stdout=logfile)
            self.build_from_sumo()
            self.tls_ids = traci.trafficlight.getIDList()

            self.logger.info(f"Simulation started on port {self.port}. Detected TLS IDs: {self.tls_ids}")
            return True
        except Exception as e:
            self.logger.error(f"Error starting SUMO on port {port}: {e}")
            return False

    def simulation_step(self) -> None:
        """Advance the simulation one step."""
        traci.simulationStep()

    def reset_simulation(self) -> None:
        """Reset the simulation to initial state."""
        # Reload the config
        traci.load([
            "-c", self.sumo_config_file,
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--time-to-teleport", "-1"
        ])

        # Rebuild network
        self.build_from_sumo()

        # Reset counters
        self.arrived_so_far = 0

    def close(self) -> None:
        """Close the simulation."""
        traci.close()

    def is_simulation_complete(self) -> bool:
        """Check if the simulation is complete.

        Returns:
            True if no more vehicles are expected, False otherwise
        """
        try:
            return traci.simulation.getMinExpectedNumber() <= 0
        except Exception as e:
            self.logger.error(f"Error checking if simulation is complete: {e}")
            return True  # Assume simulation ended on error

    def update_arrivals(self) -> None:
        """Update count of arrived vehicles."""
        arrived_ids = traci.simulation.getArrivedIDList()
        self.arrived_so_far += len(arrived_ids)

    def get_departed_vehicles_count(self) -> int:
        """Get count of departed vehicles."""
        return traci.simulation.getDepartedNumber()

    def get_arrived_vehicles_count(self) -> int:
        """Get count of arrived vehicles."""
        return self.arrived_so_far

    def get_current_time(self) -> float:
        """Get current simulation time."""
        return traci.simulation.getTime()
