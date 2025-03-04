import os
import json
import logging


class NetworkExporter:
    """Helper class for exporting network data for visualization."""

    @staticmethod
    def export_network_data(network, network_name):
        """Export network data as JSON files for visualization.

        Args:
            network: Network object containing the graph
            network_name: Name of the network (used for output filename)

        Returns:
            Tuple of (nodes_file_path, edges_file_path, geojson_file_path)
        """
        # Export traditional JSON files
        nodes_file, edges_file = NetworkExporter.export_json_files(network, network_name)

        # Export GeoJSON format (better for map visualization)
        geojson_file = NetworkExporter.export_geojson(network, network_name)

        return nodes_file, edges_file, geojson_file

    @staticmethod
    def export_json_files(network, network_name):
        """Export network data as separate JSON files for nodes and edges.

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
            return nodes_file, edges_file

        except Exception as e:
            logger.error(f"Error exporting network data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None

    @staticmethod
    def export_geojson(network, network_name):
        """Export network data as GeoJSON for visualization.

        Args:
            network: Network object containing the graph
            network_name: Name of the network (used for output filename)

        Returns:
            Path to the exported GeoJSON file
        """
        logger = logging.getLogger('NetworkExporter')

        # Set up file path
        export_dir = os.path.join('data', 'visualization')
        os.makedirs(export_dir, exist_ok=True)

        geojson_file = os.path.join(export_dir, f'{network_name}_network.geojson')

        # Check if file already exists
        if os.path.exists(geojson_file):
            logger.info(f"GeoJSON for {network_name} already exists, skipping export")
            return geojson_file

        logger.info(f"Exporting GeoJSON for {network_name}")

        try:
            # Create GeoJSON structure
            geojson = {
                "type": "FeatureCollection",
                "features": []
            }

            # Extract nodes with geographic coordinates
            import traci

            # Keep track of tls_ids for marking traffic lights
            tls_ids = set(network.tls_ids)

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

                    if lon is not None and lat is not None:
                        # Create GeoJSON Point feature
                        feature = {
                            "type": "Feature",
                            "geometry": {
                                "type": "Point",
                                "coordinates": [lon, lat]
                            },
                            "properties": {
                                "id": n,
                                "type": data.get('type', 'junction'),
                                "is_tls": n in tls_ids,
                                "x": float(x),
                                "y": float(y)
                            }
                        }

                        # Add any additional properties
                        for key, value in data.items():
                            if isinstance(value, (str, int, float, bool, type(None))):
                                feature["properties"][key] = value

                        geojson["features"].append(feature)

                except Exception as e:
                    logger.warning(f"Could not process node {n}: {e}")

            # Extract edges as LineStrings
            for u, v, data in network.graph.edges(data=True):
                try:
                    # Get source and target nodes
                    source_coords = None
                    target_coords = None

                    # Get coordinates for both nodes
                    try:
                        source_x, source_y = traci.junction.getPosition(u)
                        target_x, target_y = traci.junction.getPosition(v)

                        # Convert to geographic coordinates
                        source_lon, source_lat = traci.simulation.convertGeo(source_x, source_y)
                        target_lon, target_lat = traci.simulation.convertGeo(target_x, target_y)

                        source_coords = [source_lon, source_lat]
                        target_coords = [target_lon, target_lat]
                    except Exception as geo_error:
                        logger.warning(f"Could not get coordinates for edge {u}-{v}: {geo_error}")
                        continue

                    # Create GeoJSON LineString feature
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [source_coords, target_coords]
                        },
                        "properties": {
                            "id": data.get('id', f"{u}_{v}"),
                            "from": u,
                            "to": v,
                            "connects_tls": u in tls_ids or v in tls_ids
                        }
                    }

                    # Add any additional properties
                    for key, value in data.items():
                        if isinstance(value, (str, int, float, bool, type(None))):
                            feature["properties"][key] = value

                    geojson["features"].append(feature)

                except Exception as e:
                    logger.warning(f"Could not process edge {u}-{v}: {e}")

            # Write to file
            with open(geojson_file, 'w') as f:
                json.dump(geojson, f, indent=2)

            logger.info(f"GeoJSON exported successfully to {geojson_file}")
            return geojson_file

        except Exception as e:
            logger.error(f"Error exporting GeoJSON: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None