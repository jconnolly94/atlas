import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
from pathlib import Path
import os

# ---------- Initialize Dash app ----------
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
app.title = "Traffic Signal Control Dashboard"
server = app.server


# ---------- Data Loading ----------
# ---------- Data Loading ----------
def load_data(network_name=None):
    """Load and preprocess all data for the dashboard"""
    # Define file paths
    data_dir = Path("../src/data/datastore")
    step_data_path = data_dir / "steps/combined_step_data.csv"
    episode_data_path = data_dir / "episodes/combined_episode_data.csv"

    # Load CSV data
    step_df = pd.read_csv(step_data_path)
    episode_df = pd.read_csv(episode_data_path)

    # Get all available networks
    available_networks = sorted(step_df['network'].unique())

    # Filter by network_name if provided
    if network_name:
        step_df = step_df[step_df['network'] == network_name]
        episode_df = episode_df[episode_df['network'] == network_name]

        # Try to load network-specific nodes and edges
        vis_dir = Path("../src/data/visualization")
        nodes_path = vis_dir / f"{network_name}_nodes.json"
        edges_path = vis_dir / f"{network_name}_edges.json"
    else:
        # Default network file paths
        nodes_path = Path("../src/network_nodes.json")
        edges_path = Path("../src/network_edges.json")

    # Load network data
    try:
        with open(nodes_path, 'r') as f:
            nodes = json.load(f)
        with open(edges_path, 'r') as f:
            edges = json.load(f)
    except FileNotFoundError:
        # Try alternative paths
        try:
            print(f"Network files for {network_name} not found at {nodes_path}, trying alternatives")
            if network_name:
                # Try with DublinRd as fallback (since we have this file)
                alt_nodes_path = Path("../src/DublinRd_nodes.json")
                alt_edges_path = Path("../src/DublinRd_edges.json")
            else:
                alt_nodes_path = Path("../src/network_nodes.json")
                alt_edges_path = Path("../src/network_edges.json")

            with open(alt_nodes_path, 'r') as f:
                nodes = json.load(f)
            with open(alt_edges_path, 'r') as f:
                edges = json.load(f)
            print(f"Using alternative network files: {alt_nodes_path}, {alt_edges_path}")
        except FileNotFoundError:
            # Provide empty defaults
            print(f"Network files not found, using empty defaults")
            nodes = []
            edges = []

    # Create network DataFrame from nodes and edges
    nodes_df = pd.DataFrame(nodes)
    edges_df = pd.DataFrame(edges)

    # Get unique values for filters
    agent_types = sorted(step_df['agent_type'].unique())
    episodes = sorted(step_df['episode'].unique())
    tls_ids = sorted(step_df['tls_id'].unique())

    # Calculate aggregated data for performance metrics
    agent_summary = episode_df.groupby('agent_type').agg({
        'avg_waiting': ['mean', 'std', 'min', 'max'],
        'arrived_vehicles': ['sum', 'mean'],
        'total_reward': ['mean', 'std', 'min', 'max']
    }).reset_index()

    # Flatten multi-level column names
    agent_summary.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agent_summary.columns]

    # Calculate learning improvement (first vs last episode)
    learning_data = []
    for agent in agent_types:
        agent_episodes = episode_df[episode_df['agent_type'] == agent].sort_values('episode')
        if len(agent_episodes) >= 2:
            first_ep = agent_episodes.iloc[0]
            last_ep = agent_episodes.iloc[-1]
            waiting_improvement = first_ep['avg_waiting'] - last_ep['avg_waiting']
            waiting_pct = (1 - last_ep['avg_waiting'] / first_ep['avg_waiting']) * 100 if first_ep[
                                                                                              'avg_waiting'] > 0 else 0
            reward_improvement = last_ep['total_reward'] - first_ep['total_reward']
            reward_pct = (last_ep['total_reward'] / first_ep['total_reward'] - 1) * 100 if first_ep[
                                                                                               'total_reward'] != 0 else 0

            learning_data.append({
                'agent_type': agent,
                'waiting_time_improvement': waiting_improvement,
                'waiting_time_improvement_pct': waiting_pct,
                'reward_improvement': reward_improvement,
                'reward_improvement_pct': reward_pct
            })

    learning_df = pd.DataFrame(learning_data)

    # Calculate junction performance
    junction_perf = step_df.groupby(['agent_type', 'tls_id']).agg({
        'waiting_time': 'mean',
        'queue_length': 'mean',
        'reward': 'mean',
        'vehicle_count': 'mean'
    }).reset_index()

    # Identify critical junctions (highest waiting times)
    critical_junctions = junction_perf.groupby('tls_id')['waiting_time'].mean().reset_index()
    critical_junctions = critical_junctions.sort_values('waiting_time', ascending=False)

    # Map TLS IDs to node IDs if possible
    tls_to_node = {}
    for node in nodes:
        # Assume TLS IDs match node IDs in some cases
        if node['id'] in tls_ids:
            tls_to_node[node['id']] = node

    return {
        'step_df': step_df,
        'episode_df': episode_df,
        'nodes': nodes,
        'edges': edges,
        'nodes_df': nodes_df,
        'edges_df': edges_df,
        'available_networks': available_networks,
        'current_network': network_name,
        'agent_types': agent_types,
        'episodes': episodes,
        'tls_ids': tls_ids,
        'agent_summary': agent_summary,
        'learning_df': learning_df,
        'junction_perf': junction_perf,
        'critical_junctions': critical_junctions,
        'tls_to_node': tls_to_node
    }


# Load all data initially without filtering
data = load_data()

# Get available networks
available_networks = data['available_networks']

# If there are available networks, reload the data with the first network
if available_networks:
    data = load_data(available_networks[0])

# ---------- App Layout ----------
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Traffic Signal Control Dashboard", className="app-header-title"),
        html.P("Analysis of Reinforcement Learning Traffic Control Performance", className="app-header-subtitle")
    ], className="app-header"),

    # Network selector (add this new section)
    html.Div([
        html.Label("Select Network:"),
        dcc.Dropdown(
            id="network-dropdown",
            options=[{'label': net, 'value': net} for net in available_networks],
            value=available_networks[0] if available_networks else None,
            clearable=False,
            className="network-selector-dropdown"
        )
    ], className="network-selector"),

    # Main tabs
    dcc.Tabs(id="tabs", value="overview", children=[
        dcc.Tab(label="Overview", value="overview"),
        dcc.Tab(label="Geographic View", value="geographic"),
        dcc.Tab(label="Learning Analysis", value="learning"),
        dcc.Tab(label="Detailed Analysis", value="detailed"),
        dcc.Tab(label="Key Insights", value="insights")
    ], className="app-tabs"),

    # Tab content will be rendered here
    html.Div(id="tab-content", className="app-tab-content")
], className="app-container")


# ---------- Tab Content Callbacks ----------
@callback(
    Output("tab-content", "children"),
    [Input("tabs", "value"),
     Input("network-dropdown", "value")]
)
def render_tab_content(tab, network):
    """Render the appropriate tab content based on the selected tab and network"""
    global data

    # Update data if network has changed
    if network and network != data.get('current_network'):
        data = load_data(network)

    if tab == "overview":
        return overview_layout()
    elif tab == "geographic":
        return geographic_layout()
    elif tab == "learning":
        return learning_analysis_layout()
    elif tab == "detailed":
        return detailed_analysis_layout()
    elif tab == "insights":
        return insights_layout()
    return html.Div("Tab content not found")


# ---------- Overview Tab ----------
def overview_layout():
    """Layout for the Overview tab"""
    # Calculate summary stats for KPI boxes
    avg_waiting = data['agent_summary']['avg_waiting_mean'].mean()
    total_vehicles = data['agent_summary']['arrived_vehicles_sum'].sum()
    best_agent = data['agent_summary'].loc[data['agent_summary']['avg_waiting_mean'].idxmin()]['agent_type']

    return html.Div([
        # Filters for Overview tab
        html.Div([
            html.H3("Overview", className="tab-title"),
            html.P("High-level performance metrics across all agents and episodes", className="tab-description")
        ], className="tab-header"),

        # KPI Cards
        html.Div([
            html.Div([
                html.H4("Average Waiting Time"),
                html.Div([
                    html.Span(f"{avg_waiting:.2f}", className="metric-value"),
                    html.Span("seconds", className="metric-unit")
                ], className="metric-display"),
                html.P("Across all agents and episodes", className="metric-description")
            ], className="metric-card"),

            html.Div([
                html.H4("Total Vehicles Processed"),
                html.Div([
                    html.Span(f"{total_vehicles:,}", className="metric-value"),
                    html.Span("vehicles", className="metric-unit")
                ], className="metric-display"),
                html.P("Successfully arrived at destination", className="metric-description")
            ], className="metric-card"),

            html.Div([
                html.H4("Best Performing Agent"),
                html.Div([
                    html.Span(best_agent, className="metric-value best-agent")
                ], className="metric-display"),
                html.P("Based on average waiting time", className="metric-description")
            ], className="metric-card")
        ], className="metric-container"),

        # Charts
        html.Div([
            html.Div([
                html.H4("Agent Performance Comparison"),
                dcc.Graph(
                    id='agent-performance-chart',
                    figure=create_agent_performance_chart(data['agent_summary']),
                    config={'displayModeBar': False}
                )
            ], className="chart-container half-width"),

            html.Div([
                html.H4("Total Reward by Agent"),
                dcc.Graph(
                    id='agent-reward-chart',
                    figure=create_agent_reward_chart(data['agent_summary']),
                    config={'displayModeBar': False}
                )
            ], className="chart-container half-width")
        ], className="chart-row"),

        html.Div([
            html.Div([
                html.H4("Learning Improvement"),
                dcc.Graph(
                    id='learning-improvement-chart',
                    figure=create_learning_improvement_chart(data['learning_df']),
                    config={'displayModeBar': False}
                )
            ], className="chart-container")
        ], className="chart-row")
    ])


# ---------- Geographic View Tab ----------
def geographic_layout():
    """Layout for the Geographic View tab"""
    return html.Div([
        # Filters for Geographic View
        html.Div([
            html.Div([
                html.H3("Geographic Traffic View", className="tab-title"),
                html.P("Visualize traffic conditions and junction performance on a map", className="tab-description"),
            ], className="tab-header-left"),

            html.Div([
                html.Div([
                    html.Label("Agent Type:"),
                    dcc.Dropdown(
                        id='geo-agent-dropdown',
                        options=[{'label': agent, 'value': agent} for agent in data['agent_types']],
                        value=data['agent_types'][0],
                        clearable=False,
                        className="filter-dropdown"
                    )
                ], className="filter-item"),

                html.Div([
                    html.Label("Episode:"),
                    dcc.Dropdown(
                        id='geo-episode-dropdown',
                        options=[{'label': f'Episode {ep}', 'value': ep} for ep in data['episodes']],
                        value=data['episodes'][0],
                        clearable=False,
                        className="filter-dropdown"
                    )
                ], className="filter-item"),

                html.Div([
                    html.Label("Time Step:"),
                    dcc.Slider(
                        id='geo-timestep-slider',
                        min=0,
                        max=100,  # Will be updated by callback
                        step=1,
                        value=0,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], className="filter-item time-slider"),

                html.Div([
                    html.Button("Play", id="geo-play-button", className="play-button"),
                    html.Div(id="geo-timestep-display", className="timestep-display")
                ], className="playback-controls")
            ], className="tab-header-right")
        ], className="tab-header geo-header"),

        # Map and Stats
        html.Div([
            html.Div([
                dcc.Graph(
                    id='traffic-map',
                    figure=create_traffic_map(data),
                    config={'displayModeBar': True}
                )
            ], className="chart-container map-container"),

            html.Div([
                html.Div([
                    html.H4("Traffic Metrics"),
                    html.Div(id="selected-junction-details", className="junction-details"),

                    html.H5("Waiting Time Distribution"),
                    dcc.Graph(
                        id='waiting-time-heatmap',
                        figure=create_empty_heatmap(),
                        config={'displayModeBar': False}
                    ),

                    html.H5("Queue Length Distribution"),
                    dcc.Graph(
                        id='queue-length-bar',
                        figure=create_empty_bar(),
                        config={'displayModeBar': False}
                    )
                ], className="metrics-panel")
            ], className="chart-container metrics-container")
        ], className="chart-row map-row"),

        # Hidden div for storing map click data
        html.Div(id='selected-junction', style={'display': 'none'}),

        # Interval component for animation
        dcc.Interval(
            id='animation-interval',
            interval=500,  # in milliseconds
            n_intervals=0,
            disabled=True
        )
    ])


# ---------- Learning Analysis Tab ----------
def learning_analysis_layout():
    """Layout for the Learning Analysis tab"""
    return html.Div([
        # Filters for Learning Analysis
        html.Div([
            html.Div([
                html.H3("Learning Analysis", className="tab-title"),
                html.P("Track how agents learn and improve over episodes", className="tab-description")
            ], className="tab-header-left"),

            html.Div([
                html.Label("Compare Agents:"),
                dcc.Dropdown(
                    id='learning-agent-dropdown',
                    options=[{'label': agent, 'value': agent} for agent in data['agent_types']],
                    value=data['agent_types'],
                    multi=True,
                    className="filter-dropdown"
                )
            ], className="tab-header-right")
        ], className="tab-header"),

        # Progression Charts
        html.Div([
            html.Div([
                html.H4("Waiting Time Progression"),
                dcc.Graph(
                    id='waiting-progression-chart',
                    figure=create_empty_line(),
                    config={'displayModeBar': False}
                )
            ], className="chart-container half-width"),

            html.Div([
                html.H4("Reward Progression"),
                dcc.Graph(
                    id='reward-progression-chart',
                    figure=create_empty_line(),
                    config={'displayModeBar': False}
                )
            ], className="chart-container half-width")
        ], className="chart-row"),

        html.Div([
            html.Div([
                html.H4("Learning Rate Comparison"),
                dcc.Graph(
                    id='learning-rate-chart',
                    figure=create_learning_rate_chart(data['learning_df']),
                    config={'displayModeBar': False}
                )
            ], className="chart-container half-width"),

            html.Div([
                html.H4("Action Selection Distribution"),
                # Filters for action distribution
                html.Div([
                    html.Label("Agent:"),
                    dcc.Dropdown(
                        id='action-agent-dropdown',
                        options=[{'label': agent, 'value': agent} for agent in data['agent_types']],
                        value=data['agent_types'][0],
                        clearable=False,
                        className="filter-dropdown"
                    )
                ], className="chart-filter"),
                dcc.Graph(
                    id='action-distribution-chart',
                    figure=create_empty_bar(),
                    config={'displayModeBar': False}
                )
            ], className="chart-container half-width")
        ], className="chart-row")
    ])


# ---------- Detailed Analysis Tab ----------
def detailed_analysis_layout():
    """Layout for the Detailed Analysis tab"""
    return html.Div([
        # Filters for Detailed Analysis
        html.Div([
            html.Div([
                html.H3("Detailed Analysis", className="tab-title"),
                html.P("In-depth performance metrics for specific scenarios", className="tab-description")
            ], className="tab-header-left"),

            html.Div([
                html.Div([
                    html.Label("Agent:"),
                    dcc.Dropdown(
                        id='detailed-agent-dropdown',
                        options=[{'label': agent, 'value': agent} for agent in data['agent_types']],
                        value=data['agent_types'][0],
                        clearable=False,
                        className="filter-dropdown"
                    )
                ], className="filter-item"),

                html.Div([
                    html.Label("Traffic Light:"),
                    dcc.Dropdown(
                        id='detailed-tls-dropdown',
                        options=[{'label': tls, 'value': tls} for tls in data['tls_ids']],
                        value=data['tls_ids'][0],
                        clearable=False,
                        className="filter-dropdown"
                    )
                ], className="filter-item"),

                html.Div([
                    html.Label("Episode:"),
                    dcc.Dropdown(
                        id='detailed-episode-dropdown',
                        options=[{'label': f'Episode {ep}', 'value': ep} for ep in data['episodes']],
                        value=data['episodes'][0],
                        clearable=False,
                        className="filter-dropdown"
                    )
                ], className="filter-item")
            ], className="tab-header-right")
        ], className="tab-header"),

        # Step Analysis
        html.Div([
            html.Div([
                html.H4("Step-by-Step Analysis"),
                html.Div([
                    html.Label("Metric:"),
                    dcc.Dropdown(
                        id='metric-dropdown',
                        options=[
                            {'label': 'Waiting Time', 'value': 'waiting_time'},
                            {'label': 'Queue Length', 'value': 'queue_length'},
                            {'label': 'Reward', 'value': 'reward'},
                            {'label': 'Vehicle Count', 'value': 'vehicle_count'}
                        ],
                        value='waiting_time',
                        clearable=False,
                        className="filter-dropdown"
                    )
                ], className="chart-filter"),
                dcc.Graph(
                    id='step-analysis-chart',
                    figure=create_empty_line(),
                    config={'displayModeBar': False}
                )
            ], className="chart-container")
        ], className="chart-row"),

        # TLS Performance and Phase Distribution
        html.Div([
            html.Div([
                html.H4("TLS Performance by Episode"),
                dcc.Graph(
                    id='tls-episode-chart',
                    figure=create_empty_line(),
                    config={'displayModeBar': False}
                )
            ], className="chart-container half-width"),

            html.Div([
                html.H4("Phase Distribution"),
                dcc.Graph(
                    id='phase-distribution-chart',
                    figure=create_empty_bar(),
                    config={'displayModeBar': False}
                )
            ], className="chart-container half-width")
        ], className="chart-row")
    ])


# ---------- Key Insights Tab ----------
def insights_layout():
    """Layout for the Key Insights tab"""

    # Generate insights
    insights = generate_insights(data)

    return html.Div([
        # Insights Header
        html.Div([
            html.H3("Key Insights", className="tab-title"),
            html.P("Automatically generated findings and patterns from the data", className="tab-description")
        ], className="tab-header"),

        # Key Findings Box
        html.Div([
            html.H4("Key Findings", className="insights-title"),
            html.Div([
                html.Div([
                    html.H5(insight['title'], className=f"insight-title {insight['type']}"),
                    html.P(insight['description'], className="insight-description"),
                    html.Div([
                        html.Span(str(insight['key_metric']), className="insight-metric"),
                        html.Span(insight['unit'], className="insight-unit")
                    ], className="insight-metric-container") if 'key_metric' in insight else None
                ], className=f"insight-card {insight['type']}")
                for insight in insights
            ], className="insights-container")
        ], className="insights-section"),

        # Critical Traffic Light Analysis
        html.Div([
            html.Div([
                html.H4("Critical Traffic Light Analysis"),
                dcc.Graph(
                    id='critical-tls-chart',
                    figure=create_critical_tls_chart(data),
                    config={'displayModeBar': False}
                )
            ], className="chart-container half-width"),

            html.Div([
                html.H4("Agent Effectiveness Comparison"),
                dcc.Graph(
                    id='agent-effectiveness-chart',
                    figure=create_agent_effectiveness_chart(data),
                    config={'displayModeBar': False}
                )
            ], className="chart-container half-width")
        ], className="chart-row"),

        # Comparative Analysis
        html.Div([
            html.Div([
                html.H4("Comparative Analysis by Junction"),
                dcc.Graph(
                    id='comparative-analysis-chart',
                    figure=create_comparative_analysis_chart(data),
                    config={'displayModeBar': False}
                )
            ], className="chart-container")
        ], className="chart-row")
    ])


# ---------- Chart Creation Functions ----------
def create_agent_performance_chart(agent_summary):
    """Create agent performance comparison chart"""
    fig = px.bar(
        agent_summary,
        x='agent_type',
        y='avg_waiting_mean',
        error_y='avg_waiting_std',
        labels={'agent_type': 'Agent', 'avg_waiting_mean': 'Average Waiting Time (s)'},
        color_discrete_sequence=['#4299e1']
    )

    # Add value labels on top of bars
    for i, row in enumerate(agent_summary.itertuples()):
        fig.add_annotation(
            x=row.agent_type,
            y=row.avg_waiting_mean,
            text=f"{row.avg_waiting_mean:.1f}",
            showarrow=False,
            yshift=10
        )

    # Update layout
    fig.update_layout(
        margin=dict(l=40, r=40, t=10, b=40),
        xaxis_title="Agent Type",
        yaxis_title="Average Waiting Time (s)",
        plot_bgcolor="white",
        height=350
    )

    # Update axes
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(gridcolor='lightgray')

    return fig


def create_agent_reward_chart(agent_summary):
    """Create agent reward chart"""
    fig = px.bar(
        agent_summary,
        x='agent_type',
        y='total_reward_mean',
        error_y='total_reward_std',
        labels={'agent_type': 'Agent', 'total_reward_mean': 'Average Total Reward'},
        color_discrete_sequence=['#48bb78']
    )

    # Add value labels on top of bars
    for i, row in enumerate(agent_summary.itertuples()):
        fig.add_annotation(
            x=row.agent_type,
            y=row.total_reward_mean,
            text=f"{row.total_reward_mean:.1f}",
            showarrow=False,
            yshift=10
        )

    # Add zero line
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(agent_summary) - 0.5,
        y0=0,
        y1=0,
        line=dict(color="black", width=1, dash="dash")
    )

    # Update layout
    fig.update_layout(
        margin=dict(l=40, r=40, t=10, b=40),
        xaxis_title="Agent Type",
        yaxis_title="Average Total Reward",
        plot_bgcolor="white",
        height=350
    )

    # Update axes
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(gridcolor='lightgray')

    return fig


def create_learning_improvement_chart(learning_df):
    """Create learning improvement chart"""
    fig = px.bar(
        learning_df,
        x='agent_type',
        y=['waiting_time_improvement_pct', 'reward_improvement_pct'],
        barmode='group',
        labels={
            'agent_type': 'Agent',
            'value': 'Improvement (%)',
            'variable': 'Metric'
        },
        color_discrete_map={
            'waiting_time_improvement_pct': '#4299e1',
            'reward_improvement_pct': '#48bb78'
        }
    )

    # Update legend labels
    fig.for_each_trace(lambda t: t.update(
        name='Waiting Time' if t.name == 'waiting_time_improvement_pct' else 'Reward'
    ))

    # Add value labels on top of bars
    for i, row in enumerate(learning_df.itertuples()):
        fig.add_annotation(
            x=row.agent_type,
            y=row.waiting_time_improvement_pct,
            text=f"{row.waiting_time_improvement_pct:.1f}%",
            showarrow=False,
            xshift=-15,
            yshift=10
        )

        fig.add_annotation(
            x=row.agent_type,
            y=row.reward_improvement_pct,
            text=f"{row.reward_improvement_pct:.1f}%",
            showarrow=False,
            xshift=15,
            yshift=10
        )

    # Update layout
    fig.update_layout(
        margin=dict(l=40, r=40, t=10, b=40),
        xaxis_title="Agent Type",
        yaxis_title="Improvement (%)",
        plot_bgcolor="white",
        height=350,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update axes
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(gridcolor='lightgray')

    return fig


def create_traffic_map(data):
    """Create geographic traffic map"""
    # Check if lat/lon coordinates are available
    has_geo = all(('lat' in node and 'lon' in node and node['lat'] is not None and node['lon'] is not None)
                  for node in data['nodes'])

    # Print coordinates for debugging
    print(f"Using geographic coordinates: {has_geo}")
    if has_geo:
        for i, node in enumerate(data['nodes'][:5]):  # Print first 5 nodes
            print(f"Node {i}: {node.get('id')}, lat={node.get('lat')}, lon={node.get('lon')}")

    if has_geo:
        # Prepare data for map with actual coordinates
        nodes_df = pd.DataFrame(data['nodes'])

        # Convert lat/lon to numeric if they're strings
        for col in ['lat', 'lon']:
            if nodes_df[col].dtype == 'object':
                nodes_df[col] = pd.to_numeric(nodes_df[col], errors='coerce')

        nodes_df['is_tls'] = nodes_df['id'].isin(data['tls_ids'])

        # Remove points with invalid coordinates
        valid_coords = (~nodes_df['lat'].isna() & ~nodes_df['lon'].isna() &
                        (nodes_df['lat'] != 0) & (nodes_df['lon'] != 0))

        if valid_coords.sum() == 0:
            print("No valid geographic coordinates found, falling back to network graph")
            has_geo = False
        else:
            # Filter to only valid coordinates
            nodes_df = nodes_df[valid_coords].copy()

            # Calculate center and bounds
            center_lat = nodes_df['lat'].mean()
            center_lon = nodes_df['lon'].mean()

            # Calculate appropriate zoom level based on coordinate spread
            lat_range = nodes_df['lat'].max() - nodes_df['lat'].min()
            lon_range = nodes_df['lon'].max() - nodes_df['lon'].min()

            # Determine zoom level - lower value = zoomed out more
            # Typical values: 0 (world), 5 (continent), 10 (city), 15 (streets), 20 (buildings)
            zoom_level = 10  # Default city level

            if lat_range > 0 or lon_range > 0:
                # Calculate a better zoom level based on the coordinate range
                # This is a heuristic that works reasonably well
                max_range = max(lat_range, lon_range)
                if max_range > 1:  # Country/region scale
                    zoom_level = 5
                elif max_range > 0.1:  # City scale
                    zoom_level = 10
                elif max_range > 0.01:  # Neighborhood scale
                    zoom_level = 13
                else:  # Street scale
                    zoom_level = 15

            print(f"Map center: {center_lat}, {center_lon}, zoom: {zoom_level}")

            # Create node markers - IMPORTANT: Use Scattermapbox not Scattermap
            node_markers = go.Scattermapbox(
                lat=nodes_df['lat'],
                lon=nodes_df['lon'],
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=['blue' if is_tls else 'gray' for is_tls in nodes_df['is_tls']],
                    opacity=0.8
                ),
                text=nodes_df['id'],
                textposition="top center",
                hoverinfo='text',
                name='Junctions'
            )

            # Add line traces for the edges if we have the data
            edge_traces = []
            for _, edge in data['edges_df'].iterrows():
                # Get source and target nodes
                source = nodes_df[nodes_df['id'] == edge['from']]
                target = nodes_df[nodes_df['id'] == edge['to']]

                if len(source) > 0 and len(target) > 0:
                    edge_trace = go.Scattermapbox(
                        lat=[source.iloc[0]['lat'], target.iloc[0]['lat']],
                        lon=[source.iloc[0]['lon'], target.iloc[0]['lon']],
                        mode='lines',
                        line=dict(width=3, color='gray'),
                        opacity=0.8,
                        hoverinfo='none'
                    )
                    edge_traces.append(edge_trace)

            # Combine all traces
            traces = edge_traces + [node_markers]

            # Create mapbox layout
            layout = go.Layout(
                mapbox=dict(
                    style="open-street-map",
                    center=dict(
                        lat=center_lat,
                        lon=center_lon
                    ),
                    zoom=zoom_level
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                height=600,
                showlegend=False
            )

            fig = go.Figure(data=traces, layout=layout)

    else:
        # Fall back to network graph if no valid coordinates
        has_geo = False


    if not has_geo:
        # Use x, y coordinates for a network graph instead
        nodes_df = pd.DataFrame(data['nodes'])
        edges_df = pd.DataFrame(data['edges'])

        # Check if x/y are available
        if 'x' not in nodes_df.columns or 'y' not in nodes_df.columns:
            # Generate position data if not available
            print("No x/y coordinates available, generating network layout")
            pos = {}
            for i, node_id in enumerate(nodes_df['id']):
                angle = 2 * np.pi * i / len(nodes_df)
                radius = 300
                pos[node_id] = (radius * np.cos(angle), radius * np.sin(angle))

            # Add x, y columns
            nodes_df['x'] = nodes_df['id'].map(lambda nid: pos[nid][0])
            nodes_df['y'] = nodes_df['id'].map(lambda nid: pos[nid][1])

        # Mark TLS nodes
        nodes_df['is_tls'] = nodes_df['id'].isin(data['tls_ids'])

        # Create node traces
        node_trace = go.Scatter(
            x=nodes_df['x'],
            y=nodes_df['y'],
            mode='markers+text',
            marker=dict(
                size=20,
                color=['blue' if is_tls else 'lightgray' for is_tls in nodes_df['is_tls']],
                line=dict(width=2, color='black')
            ),
            text=nodes_df['id'],
            textposition="top center",
            hoverinfo='text',
            name='Junctions'
        )

        # Create edge traces
        edge_traces = []
        for _, edge in edges_df.iterrows():
            # Find source and target nodes
            source = nodes_df[nodes_df['id'] == edge['from']]
            target = nodes_df[nodes_df['id'] == edge['to']]

            if len(source) > 0 and len(target) > 0:
                edge_trace = go.Scatter(
                    x=[source.iloc[0]['x'], target.iloc[0]['x']],
                    y=[source.iloc[0]['y'], target.iloc[0]['y']],
                    mode='lines',
                    line=dict(width=3, color='darkgray'),
                    hoverinfo='text',
                    text=edge['id'] if 'name' not in edge else edge['name'],
                )
                edge_traces.append(edge_trace)

        # Combine traces
        fig = go.Figure(data=edge_traces + [node_trace])

        # Update layout
        fig.update_layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor='white',
            height=600,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )

    return fig


def create_empty_heatmap():
    """Create empty heatmap for placeholder"""
    fig = go.Figure(data=go.Heatmap(
        z=[[0]],
        colorscale='RdYlGn_r'
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=150,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )

    return fig


def create_empty_bar():
    """Create empty bar chart for placeholder"""
    fig = go.Figure()

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=150,
        xaxis=dict(title=""),
        yaxis=dict(title="")
    )

    return fig


def create_empty_line():
    """Create empty line chart for placeholder"""
    fig = go.Figure()

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=350,
        xaxis=dict(title=""),
        yaxis=dict(title="")
    )

    return fig


def create_learning_rate_chart(learning_df):
    """Create learning rate comparison chart"""
    fig = px.bar(
        learning_df,
        x='agent_type',
        y='waiting_time_improvement_pct',
        labels={'agent_type': 'Agent', 'waiting_time_improvement_pct': 'Learning Rate (%)'},
        color_discrete_sequence=['#4299e1']
    )

    # Add value labels on top of bars
    for i, row in enumerate(learning_df.itertuples()):
        fig.add_annotation(
            x=row.agent_type,
            y=row.waiting_time_improvement_pct,
            text=f"{row.waiting_time_improvement_pct:.1f}%",
            showarrow=False,
            yshift=10
        )

    # Update layout
    fig.update_layout(
        margin=dict(l=40, r=40, t=10, b=40),
        xaxis_title="Agent Type",
        yaxis_title="Learning Rate (% Improvement)",
        plot_bgcolor="white",
        height=350
    )

    # Update axes
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(gridcolor='lightgray')

    return fig


def create_critical_tls_chart(data):
    """Create critical traffic light analysis chart"""
    # Take top 5 critical junctions
    critical_df = data['critical_junctions'].head(5)

    fig = px.bar(
        critical_df,
        x='tls_id',
        y='waiting_time',
        labels={'tls_id': 'Traffic Light ID', 'waiting_time': 'Average Waiting Time (s)'},
        color='waiting_time',
        color_continuous_scale='RdYlGn_r'
    )

    # Add value labels on top of bars
    for i, row in enumerate(critical_df.itertuples()):
        fig.add_annotation(
            x=row.tls_id,
            y=row.waiting_time,
            text=f"{row.waiting_time:.1f}",
            showarrow=False,
            yshift=10
        )

    # Update layout
    fig.update_layout(
        margin=dict(l=40, r=40, t=10, b=40),
        xaxis_title="Traffic Light ID",
        yaxis_title="Average Waiting Time (s)",
        plot_bgcolor="white",
        height=350,
        coloraxis_showscale=False
    )

    # Update axes
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(gridcolor='lightgray')

    return fig


def create_agent_effectiveness_chart(data):
    """Create agent effectiveness scatter plot"""
    # Prepare data
    plot_data = data['agent_summary'][['agent_type', 'avg_waiting_mean', 'total_reward_mean', 'arrived_vehicles_sum']]

    fig = px.scatter(
        plot_data,
        x='avg_waiting_mean',
        y='total_reward_mean',
        size='arrived_vehicles_sum',
        color='agent_type',
        text='agent_type',
        labels={
            'avg_waiting_mean': 'Average Waiting Time (s)',
            'total_reward_mean': 'Average Reward',
            'arrived_vehicles_sum': 'Vehicles Processed',
            'agent_type': 'Agent'
        }
    )

    # Update layout
    fig.update_layout(
        margin=dict(l=40, r=40, t=10, b=40),
        xaxis_title="Average Waiting Time (s)",
        yaxis_title="Average Reward",
        plot_bgcolor="white",
        height=350
    )

    # Update traces
    fig.update_traces(
        marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey')),
        textposition='top center'
    )

    # Update axes
    fig.update_xaxes(gridcolor='lightgray')
    fig.update_yaxes(gridcolor='lightgray')

    # Add zero line for reward
    fig.add_shape(
        type="line",
        x0=0,
        x1=max(plot_data['avg_waiting_mean']) * 1.1,
        y0=0,
        y1=0,
        line=dict(color="black", width=1, dash="dash")
    )

    return fig


def create_comparative_analysis_chart(data):
    """Create comparative analysis chart by junction"""
    # Take top 5 junctions and compare across agents
    top_junctions = data['critical_junctions'].head(5)['tls_id'].tolist()

    # Filter junction performance for top junctions
    filtered_data = data['junction_perf'][data['junction_perf']['tls_id'].isin(top_junctions)]

    fig = px.bar(
        filtered_data,
        x='tls_id',
        y='waiting_time',
        color='agent_type',
        barmode='group',
        labels={
            'tls_id': 'Traffic Light ID',
            'waiting_time': 'Average Waiting Time (s)',
            'agent_type': 'Agent Type'
        }
    )

    # Update layout
    fig.update_layout(
        margin=dict(l=40, r=40, t=10, b=40),
        xaxis_title="Traffic Light ID",
        yaxis_title="Average Waiting Time (s)",
        plot_bgcolor="white",
        height=350,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update axes
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(gridcolor='lightgray')

    return fig


def generate_insights(data):
    """Generate key insights from the data"""
    insights = []

    # Best agent insight
    best_agent = data['agent_summary'].loc[data['agent_summary']['avg_waiting_mean'].idxmin()]
    insights.append({
        'type': 'best_agent',
        'title': f"{best_agent['agent_type']} is the best performing agent",
        'description': f"With an average waiting time of {best_agent['avg_waiting_mean']:.2f} seconds, {best_agent['agent_type']} outperforms other agents in reducing traffic congestion.",
        'key_metric': round(best_agent['avg_waiting_mean'], 2),
        'unit': 'seconds'
    })

    # Most consistent agent insight
    most_consistent = data['agent_summary'].loc[data['agent_summary']['avg_waiting_std'].idxmin()]
    insights.append({
        'type': 'consistency',
        'title': f"{most_consistent['agent_type']} is the most consistent agent",
        'description': f"With a standard deviation of {most_consistent['avg_waiting_std']:.2f} seconds in waiting time, {most_consistent['agent_type']} shows the most stable performance across different scenarios.",
        'key_metric': round(most_consistent['avg_waiting_std'], 2),
        'unit': 'seconds (std dev)'
    })

    # Best learner insight
    if not data['learning_df'].empty:
        best_learner = data['learning_df'].loc[data['learning_df']['waiting_time_improvement_pct'].idxmax()]
        insights.append({
            'type': 'learning_rate',
            'title': f"{best_learner['agent_type']} shows the fastest learning",
            'description': f"With a {best_learner['waiting_time_improvement_pct']:.2f}% reduction in waiting time from first to last episode, {best_learner['agent_type']} demonstrates the best learning capability.",
            'key_metric': round(best_learner['waiting_time_improvement_pct'], 2),
            'unit': '% improvement'
        })

    # Critical junction insight
    if not data['critical_junctions'].empty:
        worst_junction = data['critical_junctions'].iloc[0]
        insights.append({
            'type': 'bottleneck',
            'title': f"Junction {worst_junction['tls_id']} is a critical bottleneck",
            'description': f"With an average waiting time of {worst_junction['waiting_time']:.2f} seconds, this junction experiences the most congestion and could benefit from optimization.",
            'key_metric': round(worst_junction['waiting_time'], 2),
            'unit': 'seconds'
        })

    return insights


# ---------- Interactive Callbacks ----------
# Learning Analysis Tab Callbacks
@callback(
    [Output('waiting-progression-chart', 'figure'),
     Output('reward-progression-chart', 'figure')],
    Input('learning-agent-dropdown', 'value')
)
def update_progression_charts(selected_agents):
    """Update progression charts based on selected agents"""
    if not selected_agents:
        return create_empty_line(), create_empty_line()

    # Calculate episode averages for each agent
    progression_data = []
    for agent in selected_agents:
        agent_episodes = data['episode_df'][data['episode_df']['agent_type'] == agent]
        if not agent_episodes.empty:
            agent_episodes = agent_episodes.sort_values('episode')
            for _, row in agent_episodes.iterrows():
                progression_data.append({
                    'agent_type': agent,
                    'episode': row['episode'],
                    'avg_waiting': row['avg_waiting'],
                    'total_reward': row['total_reward']
                })

    if not progression_data:
        return create_empty_line(), create_empty_line()

    progression_df = pd.DataFrame(progression_data)

    # Create waiting time progression chart
    waiting_fig = px.line(
        progression_df,
        x='episode',
        y='avg_waiting',
        color='agent_type',
        labels={
            'episode': 'Episode',
            'avg_waiting': 'Average Waiting Time (s)',
            'agent_type': 'Agent Type'
        },
        markers=True
    )

    waiting_fig.update_layout(
        margin=dict(l=40, r=40, t=10, b=40),
        xaxis_title="Episode",
        yaxis_title="Average Waiting Time (s)",
        plot_bgcolor="white",
        height=350,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    waiting_fig.update_xaxes(gridcolor='lightgray')
    waiting_fig.update_yaxes(gridcolor='lightgray')

    # Create reward progression chart
    reward_fig = px.line(
        progression_df,
        x='episode',
        y='total_reward',
        color='agent_type',
        labels={
            'episode': 'Episode',
            'total_reward': 'Total Reward',
            'agent_type': 'Agent Type'
        },
        markers=True
    )

    reward_fig.update_layout(
        margin=dict(l=40, r=40, t=10, b=40),
        xaxis_title="Episode",
        yaxis_title="Total Reward",
        plot_bgcolor="white",
        height=350,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    reward_fig.update_xaxes(gridcolor='lightgray')
    reward_fig.update_yaxes(gridcolor='lightgray')

    # Add zero line for reward
    reward_fig.add_shape(
        type="line",
        x0=min(progression_df['episode']),
        x1=max(progression_df['episode']),
        y0=0,
        y1=0,
        line=dict(color="black", width=1, dash="dash")
    )

    return waiting_fig, reward_fig


@callback(
    Output('action-distribution-chart', 'figure'),
    Input('action-agent-dropdown', 'value')
)
def update_action_distribution(selected_agent):
    """Update action distribution chart for selected agent"""
    if not selected_agent:
        return create_empty_bar()

    # Calculate action distribution
    agent_data = data['step_df'][data['step_df']['agent_type'] == selected_agent]

    # Convert action to string to handle potential tuple values
    agent_data['action_str'] = agent_data['action'].astype(str)

    action_counts = agent_data['action_str'].value_counts().reset_index()
    action_counts.columns = ['action', 'count']

    # Sort actions if they are numeric
    try:
        action_counts['action_num'] = action_counts['action'].astype(float)
        action_counts = action_counts.sort_values('action_num')
    except:
        # If not numeric, sort by string
        action_counts = action_counts.sort_values('action')

    # Create phase distribution chart
    fig = px.bar(
        action_counts,
        x='action',
        y='count',
        labels={'action': 'Action (Phase)', 'count': 'Frequency'},
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig.update_layout(
        margin=dict(l=40, r=40, t=10, b=40),
        xaxis_title="Action (Phase)",
        yaxis_title="Frequency",
        plot_bgcolor="white",
        height=300
    )

    fig.update_xaxes(gridcolor='lightgray')
    fig.update_yaxes(gridcolor='lightgray')

    return fig


# Detailed Analysis Tab Callbacks
@callback(
    Output('step-analysis-chart', 'figure'),
    [Input('detailed-agent-dropdown', 'value'),
     Input('detailed-tls-dropdown', 'value'),
     Input('detailed-episode-dropdown', 'value'),
     Input('metric-dropdown', 'value')]
)
def update_step_analysis(agent, tls, episode, metric):
    """Update step analysis chart"""
    if not agent or not tls or not episode or not metric:
        return create_empty_line()

    # Filter data
    filtered_data = data['step_df'][
        (data['step_df']['agent_type'] == agent) &
        (data['step_df']['tls_id'] == tls) &
        (data['step_df']['episode'] == episode)
        ].sort_values('step')

    if filtered_data.empty:
        return create_empty_line()

    # Get display name for metric
    metric_names = {
        'waiting_time': 'Waiting Time (s)',
        'queue_length': 'Queue Length',
        'reward': 'Reward',
        'vehicle_count': 'Vehicle Count'
    }

    metric_name = metric_names.get(metric, metric)

    # Create step analysis chart
    fig = px.line(
        filtered_data,
        x='step',
        y=metric,
        labels={'step': 'Time Step', metric: metric_name},
        markers=True
    )

    # Add moving average line
    window_size = min(10, len(filtered_data))
    if window_size > 1:
        # Create a new DataFrame to avoid the SettingWithCopyWarning
        filtered_data_with_avg = filtered_data.copy()
        filtered_data_with_avg.loc[:, 'moving_avg'] = filtered_data[metric].rolling(window=window_size).mean()

        fig.add_trace(
            go.Scatter(
                x=filtered_data_with_avg['step'],
                y=filtered_data_with_avg['moving_avg'],
                mode='lines',
                name=f'{window_size}-Step Moving Avg',
                line=dict(color='red', width=2, dash='dash')
            )
        )

    fig.update_layout(
        margin=dict(l=40, r=40, t=10, b=40),
        xaxis_title="Time Step",
        yaxis_title=metric_name,
        plot_bgcolor="white",
        height=350,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(gridcolor='lightgray')
    fig.update_yaxes(gridcolor='lightgray')

    return fig


@callback(
    Output('tls-episode-chart', 'figure'),
    [Input('detailed-agent-dropdown', 'value'),
     Input('detailed-tls-dropdown', 'value')]
)
def update_tls_episode_chart(agent, tls):
    """Update TLS performance by episode chart"""
    if not agent or not tls:
        return create_empty_line()

    # Filter data
    filtered_data = data['step_df'][
        (data['step_df']['agent_type'] == agent) &
        (data['step_df']['tls_id'] == tls)
        ]

    if filtered_data.empty:
        return create_empty_line()

    # Calculate episode averages
    episode_avgs = filtered_data.groupby('episode').agg({
        'waiting_time': 'mean',
        'reward': 'mean'
    }).reset_index()

    # Create dual-axis figure
    fig = go.Figure()

    # Add waiting time line
    fig.add_trace(
        go.Scatter(
            x=episode_avgs['episode'],
            y=episode_avgs['waiting_time'],
            mode='lines+markers',
            name='Waiting Time',
            line=dict(color='blue')
        )
    )

    # Add reward line on secondary axis
    fig.add_trace(
        go.Scatter(
            x=episode_avgs['episode'],
            y=episode_avgs['reward'],
            mode='lines+markers',
            name='Reward',
            line=dict(color='green'),
            yaxis='y2'
        )
    )

    # Update layout
    fig.update_layout(
        margin=dict(l=40, r=40, t=10, b=40),
        xaxis=dict(
            title="Episode",
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title="Waiting Time (s)",
            gridcolor='lightgray',
            side='left'
        ),
        yaxis2=dict(
            title="Reward",
            gridcolor='lightgray',
            overlaying='y',
            side='right'
        ),
        plot_bgcolor="white",
        height=350,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


@callback(
    Output('phase-distribution-chart', 'figure'),
    [Input('detailed-agent-dropdown', 'value'),
     Input('detailed-tls-dropdown', 'value')]
)
def update_phase_distribution_chart(agent, tls):
    """Update phase distribution chart"""
    if not agent or not tls:
        return create_empty_bar()

    # Filter data
    filtered_data = data['step_df'][
        (data['step_df']['agent_type'] == agent) &
        (data['step_df']['tls_id'] == tls)
        ]

    if filtered_data.empty:
        return create_empty_bar()

    # Convert action to string to handle potential tuple values
    filtered_data['action_str'] = filtered_data['action'].astype(str)

    # Calculate phase distribution
    phase_counts = filtered_data['action_str'].value_counts().reset_index()
    phase_counts.columns = ['phase', 'count']

    # Sort phases if they are numeric
    try:
        phase_counts['phase_num'] = phase_counts['phase'].astype(float)
        phase_counts = phase_counts.sort_values('phase_num')
    except:
        # If not numeric, sort by string
        phase_counts = phase_counts.sort_values('phase')

    # Create phase distribution chart
    fig = px.bar(
        phase_counts,
        x='phase',
        y='count',
        labels={'phase': 'Phase', 'count': 'Frequency'},
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig.update_layout(
        margin=dict(l=40, r=40, t=10, b=40),
        xaxis_title="Phase",
        yaxis_title="Frequency",
        plot_bgcolor="white",
        height=350
    )

    fig.update_xaxes(gridcolor='lightgray')
    fig.update_yaxes(gridcolor='lightgray')

    return fig


# Geographic View Callbacks
@callback(
    Output('geo-timestep-slider', 'max'),
    [Input('geo-agent-dropdown', 'value'),
     Input('geo-episode-dropdown', 'value')]
)
def update_timestep_slider_max(agent, episode):
    """Update the maximum value of the timestep slider"""
    if not agent or not episode:
        return 100

    # Calculate maximum step for this agent and episode
    filtered_data = data['step_df'][
        (data['step_df']['agent_type'] == agent) &
        (data['step_df']['episode'] == episode)
        ]

    if filtered_data.empty:
        return 100

    max_step = filtered_data['step'].max()
    return max_step


@callback(
    Output('geo-timestep-display', 'children'),
    Input('geo-timestep-slider', 'value')
)
def update_timestep_display(step):
    """Update the timestep display"""
    return f"Step: {step}"


@callback(
    Output('animation-interval', 'disabled'),
    Input('geo-play-button', 'n_clicks'),
    State('animation-interval', 'disabled')
)
def toggle_animation(n_clicks, current_state):
    """Toggle the animation on/off"""
    if n_clicks is None:
        return True
    return not current_state


@callback(
    Output('geo-play-button', 'children'),
    Input('animation-interval', 'disabled')
)
def update_play_button_text(disabled):
    """Update the play button text"""
    return "Play" if disabled else "Pause"


@callback(
    Output('geo-timestep-slider', 'value'),
    [Input('animation-interval', 'n_intervals'),
     Input('geo-timestep-slider', 'max')],
    [State('geo-timestep-slider', 'value')]
)
def update_timestep_on_interval(n_intervals, max_step, current_step):
    """Update the timestep slider during animation"""
    if n_intervals is None:
        return current_step

    new_step = current_step + 1
    if new_step > max_step:
        new_step = 0

    return new_step


@callback(
    [Output('traffic-map', 'figure'),
     Output('waiting-time-heatmap', 'figure'),
     Output('queue-length-bar', 'figure')],
    [Input('geo-agent-dropdown', 'value'),
     Input('geo-episode-dropdown', 'value'),
     Input('geo-timestep-slider', 'value')]
)
def update_traffic_map(agent, episode, step):
    """Update the traffic map and related visualizations"""
    if not agent or not episode:
        return create_traffic_map(data), create_empty_heatmap(), create_empty_bar()

    # Filter data for this timestep
    filtered_data = data['step_df'][
        (data['step_df']['agent_type'] == agent) &
        (data['step_df']['episode'] == episode) &
        (data['step_df']['step'] == step)
        ]

    if filtered_data.empty:
        print(f"No data for agent={agent}, episode={episode}, step={step}")
        return create_traffic_map(data), create_empty_heatmap(), create_empty_bar()

    # Update map - start with original traffic map
    map_fig = create_traffic_map(data)

    # Check which type of map we're using
    is_mapbox = any(isinstance(trace, go.Scattermapbox) for trace in map_fig.data)

    # Create a mapping from tls_id to queue_length and waiting_time
    tls_queue = dict(zip(filtered_data['tls_id'], filtered_data['queue_length']))
    tls_waiting = dict(zip(filtered_data['tls_id'], filtered_data['waiting_time']))

    # Prepare node data
    nodes_df = pd.DataFrame(data['nodes']).copy()
    nodes_df['is_tls'] = nodes_df['id'].isin(data['tls_ids'])
    nodes_df['queue_length'] = nodes_df['id'].map(lambda x: tls_queue.get(x, 0))
    nodes_df['waiting_time'] = nodes_df['id'].map(lambda x: tls_waiting.get(x, 0))

    # Create color scale for queue length
    max_queue = max(filtered_data['queue_length'].max(), 1)
    max_waiting = max(filtered_data['waiting_time'].max(), 1)

    # Normalize data for colors
    def get_color(value, max_value):
        """Get color based on normalized value from green to red"""
        # Normalize value to 0-1
        normalized = min(1.0, value / max_value) if max_value > 0 else 0

        if normalized < 0.3:
            return 'green'  # Low
        elif normalized < 0.7:
            return 'orange'  # Medium
        else:
            return 'red'  # High

    if is_mapbox:
        # Update mapbox markers
        for i, trace in enumerate(map_fig.data):
            if isinstance(trace, go.Scattermapbox) and trace.mode and 'markers' in trace.mode:
                # Get node IDs from the text property
                if trace.text is not None:
                    node_ids = trace.text

                    # Update marker colors based on queue_length for TLS nodes
                    new_colors = []
                    for node_id in node_ids:
                        if node_id in data['tls_ids']:
                            queue = tls_queue.get(node_id, 0)
                            new_colors.append(get_color(queue, max_queue))
                        else:
                            new_colors.append('gray')

                    # Update the marker colors
                    map_fig.data[i].marker.color = new_colors
    else:
        # Update network graph
        for i, trace in enumerate(map_fig.data):
            if isinstance(trace, go.Scatter) and trace.mode and 'markers' in trace.mode:
                # Get node IDs from the text property
                if trace.text is not None:
                    node_ids = trace.text

                    # Update marker colors based on queue_length for TLS nodes
                    new_colors = []
                    for node_id in node_ids:
                        if node_id in data['tls_ids']:
                            queue = tls_queue.get(node_id, 0)
                            new_colors.append(get_color(queue, max_queue))
                        else:
                            new_colors.append('lightgray')

                    # Update the marker colors
                    map_fig.data[i].marker.color = new_colors

    # Create waiting time heatmap
    tls_waiting_df = filtered_data[['tls_id', 'waiting_time']].sort_values('waiting_time', ascending=False)

    if not tls_waiting_df.empty:
        heatmap_fig = px.imshow(
            tls_waiting_df['waiting_time'].values.reshape(1, -1),
            x=tls_waiting_df['tls_id'],
            labels=dict(x='Traffic Light ID', y='', color='Waiting Time'),
            color_continuous_scale='RdYlGn_r'
        )

        heatmap_fig.update_layout(
            margin=dict(l=20, r=20, t=0, b=30),
            height=150,
            coloraxis_colorbar=dict(
                title='Waiting Time',
                thicknessmode='pixels',
                thickness=10,
                lenmode='pixels',
                len=100,
                yanchor='top',
                y=1,
                ticks='outside'
            )
        )
    else:
        heatmap_fig = create_empty_heatmap()

    # Create queue length bar chart
    tls_queue_df = filtered_data[['tls_id', 'queue_length']].sort_values('queue_length', ascending=False)

    if not tls_queue_df.empty:
        queue_fig = px.bar(
            tls_queue_df,
            x='tls_id',
            y='queue_length',
            labels={'tls_id': 'Traffic Light ID', 'queue_length': 'Queue Length'},
            color='queue_length',
            color_continuous_scale='RdYlGn_r'
        )

        queue_fig.update_layout(
            margin=dict(l=20, r=20, t=0, b=30),
            height=150,
            coloraxis_showscale=False,
            xaxis_tickangle=-45
        )
    else:
        queue_fig = create_empty_bar()

    # Add some debugging info
    print(f"Updated map for agent={agent}, episode={episode}, step={step}, TLS count={len(tls_queue_df)}")

    return map_fig, heatmap_fig, queue_fig


@callback(
    Output('selected-junction-details', 'children'),
    [Input('traffic-map', 'clickData'),
     Input('geo-agent-dropdown', 'value'),
     Input('geo-episode-dropdown', 'value'),
     Input('geo-timestep-slider', 'value')]
)
def update_selected_junction_details(click_data, agent, episode, step):
    """Update the selected junction details panel"""
    if not click_data or not agent or not episode:
        return html.P("Click on a junction to see details")

    try:
        # Extract the selected node ID
        point_index = click_data['points'][0]['pointIndex']
        selected_node = data['nodes'][point_index]
        node_id = selected_node['id']

        # Check if this node is a traffic light
        if node_id not in data['tls_ids']:
            return html.Div([
                html.P(f"Junction ID: {node_id}", className="detail-item"),
                html.P("This is not a traffic light junction", className="detail-item")
            ])

        # Get traffic data for this node
        filtered_data = data['step_df'][
            (data['step_df']['agent_type'] == agent) &
            (data['step_df']['episode'] == episode) &
            (data['step_df']['step'] == step) &
            (data['step_df']['tls_id'] == node_id)
            ]

        if filtered_data.empty:
            return html.P(f"No data available for {node_id} at step {step}")

        # Get first (should be only) row
        tls_data = filtered_data.iloc[0]

        return html.Div([
            html.P(f"Junction ID: {node_id}", className="detail-item bold"),
            html.P(f"Waiting Time: {tls_data['waiting_time']:.2f} seconds", className="detail-item"),
            html.P(f"Queue Length: {tls_data['queue_length']:.1f} vehicles", className="detail-item"),
            html.P(f"Vehicle Count: {tls_data['vehicle_count']} vehicles", className="detail-item"),
            html.P(f"Current Phase: {tls_data['action']}", className="detail-item"),
            html.P(f"Current Reward: {tls_data['reward']:.3f}", className="detail-item")
        ])

    except Exception as e:
        return html.P(f"Error: {str(e)}")


# ---------- CSS Styling ----------
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Base styles */
            body {
                font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f7fa;
                color: #333;
            }

            /* App container */
            .app-container {
                max-width: 1440px;
                margin: 0 auto;
                padding: 20px;
            }
            
            /* Network selector */
            .network-selector {
                padding: 10px 0;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                gap: 10px;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                padding: 15px;
            }
            
            .network-selector label {
                font-weight: 500;
                margin-right: 10px;
                color: #2a4365;
                min-width: 120px;
            }
            
            .network-selector-dropdown {
                min-width: 200px;
                flex-grow: 1;
                max-width: 400px;
            }

            /* Header */
            .app-header {
                text-align: center;
                margin-bottom: 20px;
            }

            .app-header-title {
                font-size: 2rem;
                margin-bottom: 5px;
                color: #2a4365;
            }

            .app-header-subtitle {
                font-size: 1rem;
                color: #4a5568;
            }

            /* Tabs */
            .app-tabs {
                margin-bottom: 20px;
            }

            /* Tab content */
            .app-tab-content {
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                padding: 20px;
            }

            /* Tab header */
            .tab-header {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: 20px;
            }

            .tab-header-left {
                flex: 1;
            }

            .tab-header-right {
                display: flex;
                flex-wrap: wrap;
                align-items: flex-end;
                gap: 15px;
            }

            .tab-title {
                font-size: 1.5rem;
                margin: 0 0 5px;
                color: #2a4365;
            }

            .tab-description {
                color: #4a5568;
                margin: 0;
            }

            /* Filters */
            .filter-item {
                min-width: 150px;
            }

            .filter-dropdown {
                width: 100%;
            }

            .chart-filter {
                margin-bottom: 10px;
                max-width: 300px;
            }

            /* Metrics */
            .metric-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }

            .metric-card {
                background: white;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                border-left: 4px solid #4299e1;
            }

            .metric-display {
                display: flex;
                align-items: baseline;
                margin: 10px 0;
            }

            .metric-value {
                font-size: 2rem;
                font-weight: bold;
                color: #2b6cb0;
            }

            .best-agent {
                color: #38a169;
            }

            .metric-unit {
                font-size: 0.9rem;
                margin-left: 5px;
                color: #718096;
            }

            .metric-description {
                color: #718096;
                font-size: 0.9rem;
                margin: 0;
            }

            /* Charts */
            .chart-row {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-bottom: 20px;
            }

            .chart-container {
                flex: 1 1 100%;
                min-width: 300px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                padding: 15px;
            }

            .half-width {
                flex: 1 1 calc(50% - 20px);
            }

            .chart-container h4 {
                margin-top: 0;
                color: #2a4365;
            }

            /* Map specific */
            .geo-header {
                flex-direction: column;
            }

            .map-row {
                flex-wrap: nowrap;
            }

            .map-container {
                flex: 3;
            }

            .metrics-container {
                flex: 1;
                min-width: 250px;
                overflow-y: auto;
                max-height: 650px;
            }

            .metrics-panel {
                height: 100%;
            }

            .time-slider {
                width: 100%;
            }

            .playback-controls {
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .play-button {
                background: #4299e1;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 4px;
                cursor: pointer;
            }

            .play-button:hover {
                background: #3182ce;
            }

            .timestep-display {
                font-weight: 500;
            }

            /* Insights */
            .insights-section {
                margin-bottom: 20px;
            }

            .insights-title {
                color: #2a4365;
                margin-top: 0;
            }

            .insights-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
            }

            .insight-card {
                padding: 15px;
                border-radius: 8px;
                background: white;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                border-left-width: 4px;
                border-left-style: solid;
            }

            .insight-card.best_agent {
                border-left-color: #38a169;
            }

            .insight-card.consistency {
                border-left-color: #3182ce;
            }

            .insight-card.learning_rate {
                border-left-color: #805ad5;
            }

            .insight-card.bottleneck {
                border-left-color: #e53e3e;
            }

            .insight-title {
                font-size: 1rem;
                margin-top: 0;
                margin-bottom: 8px;
            }

            .insight-title.best_agent {
                color: #38a169;
            }

            .insight-title.consistency {
                color: #3182ce;
            }

            .insight-title.learning_rate {
                color: #805ad5;
            }

            .insight-title.bottleneck {
                color: #e53e3e;
            }

            .insight-description {
                font-size: 0.9rem;
                margin: 0 0 10px;
                color: #4a5568;
            }

            .insight-metric-container {
                display: flex;
                align-items: baseline;
            }

            .insight-metric {
                font-size: 1.5rem;
                font-weight: bold;
                color: #2b6cb0;
            }

            .insight-unit {
                font-size: 0.8rem;
                margin-left: 5px;
                color: #718096;
            }

            /* Junction details */
            .junction-details {
                background: #f0f4f8;
                border-radius: 4px;
                padding: 10px;
                margin-bottom: 15px;
            }

            .detail-item {
                margin: 5px 0;
            }

            .bold {
                font-weight: bold;
            }

            /* Responsive adjustments */
            @media (max-width: 768px) {
                .chart-row {
                    flex-direction: column;
                }

                .half-width {
                    flex: 1 1 100%;
                }

                .map-row {
                    flex-direction: column;
                }

                .tab-header {
                    flex-direction: column;
                }

                .tab-header-right {
                    margin-top: 15px;
                    width: 100%;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)