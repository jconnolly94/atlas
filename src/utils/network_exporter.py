import os
import json
import logging
import shutil


class NetworkExporter:
    """Helper class for exporting network data for visualization."""

    @staticmethod
    def export_network_data(network, network_name):
        """Export network data as JSON files for visualization.

        Args:
            network: Network object containing the graph
            network_name: Name of the network (used for output filename)

        Returns:
            Tuple of (nodes_file_path, edges_file_path)
        """
        logger = logging.getLogger('NetworkExporter')

        # Set up file paths
        export_dir = os.path.join('data', 'visualization')
        os.makedirs(export_dir, exist_ok=True)

        nodes_file = os.path.join(export_dir, f'{network_name}_nodes.json')
        edges_file = os.path.join(export_dir, f'{network_name}_edges.json')

        # Check if files already exist
        if os.path.exists(nodes_file) and os.path.exists(edges_file):
            logger.info(f"Network data for {network_name} already exists, skipping export")

            # Copy to src directory for dashboard access
            src_dir = os.path.join('src', 'data', 'visualization')
            os.makedirs(src_dir, exist_ok=True)

            src_nodes_file = os.path.join(src_dir, f'{network_name}_nodes.json')
            src_edges_file = os.path.join(src_dir, f'{network_name}_edges.json')

            try:
                shutil.copy2(nodes_file, src_nodes_file)
                shutil.copy2(edges_file, src_edges_file)
                logger.info(f"Copied network files to src directory for dashboard")
            except Exception as copy_error:
                logger.warning(f"Could not copy network files to src directory: {copy_error}")

            return nodes_file, edges_file

        logger.info(f"Exporting network data for {network_name}")

        try:
            # Extract nodes with geographic coordinates
            nodes = []
            import traci

            for n, data in network.graph.nodes(data=True):
                try:
                    # Get position from TraCI
                    x, y = traci.junction.getPosition(n)

                    # Convert to geographic coordinates
                    try:
                        lon, lat = traci.simulation.convertGeo(x, y)
                    except Exception as geo_error:
                        logger.warning(f"Could not convert coordinates for node {n}: {geo_error}")
                        lon, lat = None, None

                except Exception as e:
                    # Fallback to stored coordinates if available
                    logger.warning(f"Could not get position for node {n}: {e}")
                    x, y = data.get('x', 0), data.get('y', 0)
                    lon, lat = None, None

                # Add node with all available coordinates
                nodes.append({
                    'id': n,
                    'x': float(x),
                    'y': float(y),
                    'lon': float(lon) if lon is not None else None,
                    'lat': float(lat) if lat is not None else None
                })

            # Extract edges with all data
            edges = []
            for u, v, data in network.graph.edges(data=True):
                # Create a copy of data without any potential non-serializable objects
                safe_data = {}
                for key, value in data.items():
                    if isinstance(value, (str, int, float, bool, type(None), list, dict)):
                        safe_data[key] = value

                edges.append({
                    'id': data.get('id', f"{u}_{v}"),
                    'from': u,
                    'to': v,
                    **safe_data
                })

            # Write to files
            with open(nodes_file, 'w') as f:
                json.dump(nodes, f, indent=2)

            with open(edges_file, 'w') as f:
                json.dump(edges, f, indent=2)

            logger.info(f"Network data exported successfully to {nodes_file} and {edges_file}")

            # Also copy to src directory for dashboard access
            src_dir = os.path.join('src', 'data', 'visualization')
            os.makedirs(src_dir, exist_ok=True)

            src_nodes_file = os.path.join(src_dir, f'{network_name}_nodes.json')
            src_edges_file = os.path.join(src_dir, f'{network_name}_edges.json')

            try:
                shutil.copy2(nodes_file, src_nodes_file)
                shutil.copy2(edges_file, src_edges_file)
                logger.info(f"Copied network files to src directory for dashboard")
            except Exception as copy_error:
                logger.warning(f"Could not copy network files to src directory: {copy_error}")

            return nodes_file, edges_file

        except Exception as e:
            logger.error(f"Error exporting network data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None