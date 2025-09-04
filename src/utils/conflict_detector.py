"""
Conflict Detector for SUMO Traffic Signal Control

This module provides utilities for detecting conflicting traffic signal states
in SUMO networks. It detects conflicts based on shared internal lanes ("via" attributes)
and builds conflict matrices/dictionaries for traffic light signals.
"""

import os
import logging
import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _find_net_file(config_file_path: str) -> str:
    """
    Parse a SUMO .sumocfg file to extract the path to the .net.xml file.
    
    Args:
        config_file_path: Path to the SUMO configuration file (.sumocfg)
        
    Returns:
        Absolute path to the .net.xml file
        
    Raises:
        FileNotFoundError: If the config file or net file doesn't exist
        ValueError: If the net file reference is not found in the config
    """
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"SUMO config file not found: {config_file_path}")
    
    logger.info(f"Finding network file from config: {config_file_path}")
    
    try:
        # First, try XML parsing approach
        try:
            tree = ET.parse(config_file_path)
            root = tree.getroot()
            
            # Find <input> section and <net-file> tag
            input_section = root.find('./input')
            if input_section is not None:
                net_file_tag = input_section.find('./net-file')
                if net_file_tag is not None:
                    relative_path = net_file_tag.get('value')
                    if relative_path:
                        # Construct absolute path based on the config file's directory
                        config_dir = os.path.dirname(os.path.abspath(config_file_path))
                        absolute_path = os.path.normpath(os.path.join(config_dir, relative_path))
                        
                        if os.path.exists(absolute_path):
                            logger.info(f"Found network file (XML parsing): {absolute_path}")
                            return absolute_path
                        else:
                            logger.warning(f"Network file not found at: {absolute_path}")
        except ET.ParseError as e:
            logger.warning(f"XML parsing failed for SUMO config file {config_file_path}: {e}")
        
        # Fallback 1: Try simple text parsing
        try:
            with open(config_file_path, 'r') as f:
                for line in f:
                    if '<net-file value="' in line:
                        # Extract path using string operations
                        start_idx = line.find('value="') + 7
                        end_idx = line.find('"', start_idx)
                        if start_idx > 7 and end_idx > start_idx:
                            relative_path = line[start_idx:end_idx]
                            config_dir = os.path.dirname(os.path.abspath(config_file_path))
                            absolute_path = os.path.normpath(os.path.join(config_dir, relative_path))
                            
                            if os.path.exists(absolute_path):
                                logger.info(f"Found network file (text parsing): {absolute_path}")
                                return absolute_path
                            else:
                                logger.warning(f"Network file not found at: {absolute_path}")
        except Exception as e:
            logger.warning(f"Text parsing failed for SUMO config file: {e}")
        
        # Fallback 2: Look for .net.xml file in the same directory
        config_dir = os.path.dirname(os.path.abspath(config_file_path))
        config_name = os.path.splitext(os.path.basename(config_file_path))[0]
        potential_net_file = os.path.join(config_dir, f"{config_name}.net.xml")
        
        if os.path.exists(potential_net_file):
            logger.info(f"Found network file (same name): {potential_net_file}")
            return potential_net_file
            
        # Fallback 3: Look for any .net.xml file in the same directory
        net_files = [f for f in os.listdir(config_dir) if f.endswith('.net.xml')]
        if net_files:
            absolute_path = os.path.join(config_dir, net_files[0])
            logger.info(f"Found network file (directory scan): {absolute_path}")
            return absolute_path
            
        # If we get here, we've exhausted our options
        raise ValueError(f"Unable to find network file for SUMO config: {config_file_path}")
        
    except Exception as e:
        logger.error(f"Error finding network file for {config_file_path}: {e}")
        raise ValueError(f"Failed to find network file for SUMO config: {e}")


def parse_net_file(net_file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse a SUMO .net.xml file to extract connection information for traffic lights.
    
    Args:
        net_file_path: Path to the .net.xml file
        
    Returns:
        Dictionary mapping traffic light IDs to their connections:
        {
            'tls_id_1': [
                {
                    'from_edge': str,
                    'to_edge': str,
                    'from_lane': int,
                    'to_lane': int,
                    'tl': str (traffic light ID),
                    'link_index': int,
                    'via': str (internal lane ID)
                },
                ...
            ],
            'tls_id_2': [...],
            ...
        }
        
    Raises:
        FileNotFoundError: If the net file doesn't exist
        ET.ParseError: If there's an error parsing the XML
    """
    if not os.path.exists(net_file_path):
        raise FileNotFoundError(f"SUMO network file not found: {net_file_path}")
    
    try:
        tree = ET.parse(net_file_path)
        root = tree.getroot()
        
        # Dictionary to store connections by traffic light ID
        tls_connections: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Find all connection elements
        for connection in root.findall('.//connection'):
            # Only interested in connections with traffic light control
            tl = connection.get('tl')
            if tl is not None:
                link_index = connection.get('linkIndex')
                via = connection.get('via')
                
                # Ensure required attributes are present
                if link_index is None or via is None:
                    logger.warning(f"Skipping connection with missing linkIndex or via: {ET.tostring(connection, encoding='unicode')}")
                    continue
                
                # Add connection info to the dictionary
                tls_connections[tl].append({
                    'from_edge': connection.get('from'),
                    'to_edge': connection.get('to'),
                    'from_lane': int(connection.get('fromLane', 0)),
                    'to_lane': int(connection.get('toLane', 0)),
                    'tl': tl,
                    'link_index': int(link_index),
                    'via': via
                })
        
        logger.info(f"Parsed {sum(len(conns) for conns in tls_connections.values())} connections "
                   f"for {len(tls_connections)} traffic lights from {net_file_path}")
        
        return dict(tls_connections)
        
    except ET.ParseError as e:
        logger.error(f"Error parsing network file {net_file_path}: {e}")
        raise


def identify_conflicts_via(tls_connections: Dict[str, List[Dict[str, Any]]]) -> Dict[str, np.ndarray]:
    """
    Identify conflicting movements based on shared internal lanes ("via" attributes).
    
    Args:
        tls_connections: Dictionary of connections by traffic light ID (from parse_net_file)
        
    Returns:
        Dictionary mapping traffic light IDs to conflict matrices:
        {
            'tls_id_1': numpy.ndarray (n×n matrix where n is max link_index+1),
            'tls_id_2': numpy.ndarray,
            ...
        }
        
        In each matrix, conflict_matrix[i][j] = 1 if link i conflicts with link j,
        0 otherwise. The matrices are symmetric.
    """
    conflict_matrices: Dict[str, np.ndarray] = {}
    
    for tls_id, connections in tls_connections.items():
        # Skip if no connections for this TLS
        if not connections:
            logger.warning(f"No connections found for traffic light {tls_id}")
            continue
        
        # Find the maximum link index to determine matrix size
        max_link_idx = max(conn['link_index'] for conn in connections)
        num_links = max_link_idx + 1
        
        # Create conflict matrix initialized with zeros
        conflict_matrix = np.zeros((num_links, num_links), dtype=int)
        
        # Group connections by via (internal lane)
        via_to_links: Dict[str, List[int]] = defaultdict(list)
        for conn in connections:
            via = conn['via']
            link_index = conn['link_index']
            via_to_links[via].append(link_index)
        
        # Mark conflicts where links share the same via lane
        for via, link_indices in via_to_links.items():
            if len(link_indices) > 1:  # Multiple links share this internal lane
                for i in range(len(link_indices)):
                    for j in range(i + 1, len(link_indices)):
                        link1 = link_indices[i]
                        link2 = link_indices[j]
                        
                        # Mark both directions as conflicts (symmetric matrix)
                        conflict_matrix[link1][link2] = 1
                        conflict_matrix[link2][link1] = 1
        
        conflict_matrices[tls_id] = conflict_matrix
        
        logger.info(f"Created conflict matrix for TLS {tls_id} with {num_links} links")
        # Optional detailed logging for debugging
        # Log the number of conflicts detected
        num_conflicts = np.sum(conflict_matrix) // 2  # Divide by 2 due to symmetric matrix
        logger.debug(f"TLS {tls_id}: Detected {num_conflicts} conflicts among {num_links} links")
    
    return conflict_matrices


class ConflictDetector:
    """
    Detects and manages traffic signal conflicts based on SUMO network topology.
    
    The detector parses SUMO network files to identify conflicting traffic movements
    and provides utilities to check for conflicts at runtime.
    """
    
    def __init__(self, sumo_config_file: str):
        """
        Initialize the conflict detector.
        
        Args:
            sumo_config_file: Path to the SUMO configuration file (.sumocfg)
            
        Raises:
            FileNotFoundError: If the config file or net file doesn't exist
            ValueError: If there's an error parsing the config or net file
        """
        logger.info(f"Initializing ConflictDetector with config file: {sumo_config_file}")
        self.sumo_config_file = sumo_config_file
        self.conflict_dict = {}
        
        try:
            # Find the network file from the config
            self.net_file_path = _find_net_file(sumo_config_file)
            logger.info(f"Found network file: {self.net_file_path}")
            
            # Parse the network file to get connections by TLS
            self.tls_connections = parse_net_file(self.net_file_path)
            
            # Generate conflict matrices
            self.conflict_matrices = identify_conflicts_via(self.tls_connections)
            
            # Build a more convenient conflict dictionary
            self.conflict_dict = self._build_conflict_dict()
            
            logger.info(f"ConflictDetector initialized with {len(self.conflict_dict)} traffic lights")
        except Exception as e:
            logger.error(f"Error initializing ConflictDetector: {e}")
            # Initialize with empty data so methods still work (returning empty sets/false) but don't crash
            self.net_file_path = ""
            self.tls_connections = {}
            self.conflict_matrices = {}
    
    def _build_conflict_dict(self) -> Dict[str, Dict[int, Set[int]]]:
        """
        Convert conflict matrices to a nested dictionary format for efficient lookups.
        
        Returns:
            Dictionary mapping:
            tls_id -> link_index -> set of conflicting link indices
        """
        conflict_dict: Dict[str, Dict[int, Set[int]]] = {}
        
        for tls_id, matrix in self.conflict_matrices.items():
            conflict_dict[tls_id] = {}
            
            num_links = matrix.shape[0]
            for i in range(num_links):
                # Find all links that conflict with link i
                conflicting_links = set()
                for j in range(num_links):
                    if matrix[i, j] == 1:
                        conflicting_links.add(j)
                
                # Only add entry if there are any conflicts
                if conflicting_links:
                    conflict_dict[tls_id][i] = conflicting_links
        
        return conflict_dict
    
    def get_conflicting_links(self, tls_id: str, link_index: int) -> Set[int]:
        """
        Get the set of link indices that conflict with the given link.
        
        Args:
            tls_id: Traffic light ID
            link_index: Index of the link to check
            
        Returns:
            Set of link indices that conflict with the given link.
            Returns an empty set if no conflicts or if the TLS/link doesn't exist.
        """
        return self.conflict_dict.get(tls_id, {}).get(link_index, set())
    
    def check_conflict(self, tls_id: str, link1_idx: int, link2_idx: int) -> bool:
        """
        Check if two links conflict with each other.
        
        Args:
            tls_id: Traffic light ID
            link1_idx: Index of the first link
            link2_idx: Index of the second link
            
        Returns:
            True if the links conflict, False otherwise
        """
        return link2_idx in self.get_conflicting_links(tls_id, link1_idx)


# Example usage (if file is run directly)
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python conflict_detector.py <path/to/your.sumocfg>")
        sys.exit(1)
    
    # Test the conflict detector
    config_file = sys.argv[1]
    try:
        detector = ConflictDetector(config_file)
        
        # Print some stats
        print(f"Loaded conflict data for {len(detector.conflict_dict)} traffic lights")
        
        for tls_id, conflicts_by_link in detector.conflict_dict.items():
            print(f"\nTraffic Light: {tls_id}")
            print(f"Number of links with conflicts: {len(conflicts_by_link)}")
            
            total_conflicts = sum(len(conflicting) for conflicting in conflicts_by_link.values())
            print(f"Total conflicts detected: {total_conflicts}")
            
            # Print first few conflicts as example
            for link_idx, conflicting in list(conflicts_by_link.items())[:3]:
                print(f"  Link {link_idx} conflicts with: {conflicting}")
            
            if len(conflicts_by_link) > 3:
                print("  ...")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
