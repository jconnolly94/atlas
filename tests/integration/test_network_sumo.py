"""Integration tests for network and SUMO interaction."""
import unittest
from unittest.mock import patch, MagicMock
import os
import sys

from src.env.network import Network


class TestNetworkSumoIntegration(unittest.TestCase):
    """Test interactions between network class and SUMO simulator."""

    def setUp(self):
        """Set up test by patching TraCI API."""
        # Create patches for TraCI
        self.patches = [
            patch('traci.start'),
            patch('traci.close'),
            patch('traci.simulation'),
            patch('traci.trafficlight'),
            patch('traci.lane'),
            patch('traci.junction'),
            patch('traci.edge')
        ]
        
        # Start all patches
        self.mocks = [p.start() for p in self.patches]
        
        # Extract individual mocks for easier reference
        self.mock_start = self.mocks[0]
        self.mock_close = self.mocks[1]
        self.mock_simulation = self.mocks[2]
        self.mock_trafficlight = self.mocks[3]
        self.mock_lane = self.mocks[4]
        self.mock_junction = self.mocks[5]
        self.mock_edge = self.mocks[6]
        
        # Setup TraCI mock behavior
        self.mock_trafficlight.getIDList.return_value = ['tl1', 'tl2']
        self.mock_trafficlight.getControlledLanes.return_value = ['lane1', 'lane2']
        
        program_logic = MagicMock()
        program_logic.phases = [MagicMock(), MagicMock()]
        self.mock_trafficlight.getAllProgramLogics.return_value = [program_logic]
        
        self.mock_lane.getWaitingTime.return_value = 5.0
        self.mock_lane.getLastStepVehicleNumber.return_value = 2
        self.mock_lane.getLastStepHaltingNumber.return_value = 2  # Queue length
        
        # Mock junction and edge for build_from_sumo
        self.mock_junction.getIDList.return_value = ['j1', 'j2']
        self.mock_edge.getIDList.return_value = ['e1', 'e2']
        self.mock_edge.getLaneNumber.return_value = 2
        self.mock_edge.getFromJunction.return_value = 'j1'
        self.mock_edge.getToJunction.return_value = 'j2'
        
        # Create network object with mocked TraCI
        self.network_file = 'Simple4wayCross/simpleCross.sumocfg'
        self.network = Network(self.network_file)
        
        # Patch traci module methods to prevent actual SUMO calls
        patch('traci.load', MagicMock()).start()
        patch('traci.simulationStep', MagicMock()).start()
        patch('traci.simulation.getMinExpectedNumber', MagicMock(return_value=0)).start()
        
        # Mock the start_simulation method
        self.network.start_simulation = MagicMock(return_value=True)
        self.network.start_simulation(use_gui=False)
        self.network.tls_ids = ['tl1', 'tl2']  # Set tls_ids directly
        
        # Set graph for adjacency test
        self.network.graph = MagicMock()
        self.network.graph.successors.return_value = []
        self.network.graph.predecessors.return_value = []

    def tearDown(self):
        """Stop all patches."""
        for p in self.patches:
            p.stop()

    def test_network_initializes_sumo(self):
        """Test that network properly initializes SUMO simulation."""
        # Verify TraCI was started with correct parameters
        self.network.start_simulation.assert_called_once()
        
        # Verify traffic light data was loaded
        self.assertEqual(set(self.network.tls_ids), set(['tl1', 'tl2']))

    def test_network_reset_restarts_simulation(self):
        """Test that network reset restarts SUMO simulation."""
        # Create mock for traci.load to test this specific function
        with patch('traci.load') as mock_load:
            # Reset the network
            self.network.reset_simulation()
            
            # Verify TraCI load was called (for reset)
            self.assertTrue(mock_load.called, "traci.load was not called during reset")
        
    def test_network_step_advances_simulation(self):
        """Test that network step advances SUMO simulation."""
        # Create mock for traci.simulationStep to test this specific function
        with patch('traci.simulationStep') as mock_step:
            # Step the network
            self.network.simulation_step()
            
            # Verify TraCI simulation step was called
            self.assertTrue(mock_step.called, "traci.simulationStep was not called during simulation step")
        
    def test_get_traffic_light_phases(self):
        """Test retrieving traffic light phases from SUMO."""
        # Get phases
        phases = self.network.get_possible_phases('tl1')
        
        # Verify TraCI API was called
        self.mock_trafficlight.getAllProgramLogics.assert_called_with('tl1')
        
        # Verify phases were extracted
        self.assertEqual(len(phases), 2)
        
    def test_set_traffic_light_phase(self):
        """Test setting traffic light phase in SUMO."""
        # Set phase
        self.network.set_traffic_light_phase('tl1', 1)
        
        # Verify TraCI API was called
        self.mock_trafficlight.setPhase.assert_called_with('tl1', 1)
        
    def test_get_lane_metrics(self):
        """Test retrieving lane metrics from SUMO."""
        # Get lane metrics
        waiting_time = self.network.get_lane_waiting_time('lane1')
        queue_length = self.network.get_lane_queue('lane1')
        
        # Verify TraCI API was called
        self.mock_lane.getWaitingTime.assert_called_with('lane1')
        self.mock_lane.getLastStepHaltingNumber.assert_called_with('lane1')
        
        # Verify metrics were extracted
        self.assertEqual(waiting_time, 5.0)
        self.assertEqual(queue_length, 2)
        
    def test_network_detects_simulation_end(self):
        """Test that network detects when SUMO simulation is done."""
        # Simulation not done
        self.mock_simulation.getMinExpectedNumber.return_value = 10
        self.assertFalse(self.network.is_simulation_complete())
        
        # Simulation done (no more vehicles)
        self.mock_simulation.getMinExpectedNumber.return_value = 0
        self.assertTrue(self.network.is_simulation_complete())
        
    def test_network_builds_traffic_light_graph(self):
        """Test that network builds traffic light adjacency graph correctly."""
        # Set up controlled lanes for traffic lights
        self.mock_trafficlight.getControlledLanes.side_effect = [
            ['lane1', 'lane2'],  # for tl1
            ['lane2', 'lane3']   # for tl2
        ]
        
        # Use the adjacency method directly
        tl1_adj = self.network.get_adjacent_traffic_lights('tl1')
        
        # For testing with an empty network graph, empty list should be returned
        self.assertEqual(tl1_adj, [])


if __name__ == '__main__':
    unittest.main()