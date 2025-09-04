# Atlas - Adaptive Traffic Light Automation System

## Overview

Atlas is a reinforcement learning framework for intelligent traffic signal control. The system uses various machine learning approaches to optimize traffic flow by adaptively controlling traffic signals based on current traffic conditions. The project includes multiple agent implementations, traffic network configurations, and a robust simulation environment.

## Project Structure

- `/src`: Main source code directory
  - `/agents`: Various agent implementations (Q-Learning, DQN, Advanced, etc.)
  - `/data`: Data storage for simulation results
  - `/env`: Environment components including network and simulation interfaces
  - `/utils`: Utility functions and classes for data collection, network export, etc.
  - `runner.py`: Main entry point for running simulations

- `/networks`: Traffic network configurations
  - `/Simple4wayCross`: Simple intersection with four approaches
  - `/DublinRd`: Representation of Dublin Road network
  - `/RoxboroArea`: Representation of Roxboro Area network

- `/tests`: Test suite for the application

## Features

- **Lane-Level Traffic Control**: Fine-grained control over individual traffic lanes rather than entire phases
- **Multiple Agent Types**:
  - Q-Learning: Traditional reinforcement learning approach
  - DQN (Deep Q-Network): Neural network-based reinforcement learning
  - Advanced: Improved traffic control algorithm
  - Baseline: Simple rule-based approach for comparison
- **Comprehensive Metrics**: Tracking of waiting times, queue lengths, throughput, and rewards
- **Interactive Run Management**: Start new runs or resume previous experiments
- **Data Export**: Results stored in Parquet format for efficient analysis
- **Visualization Tools**: Components for data visualization and analysis

## Core Components

### Environment

The environment class (`Environment`) manages the traffic simulation with an interface to SUMO (Simulation of Urban MObility). It provides methods for:
- Getting the current state of the traffic network
- Applying actions to the traffic signals
- Calculating rewards for reinforcement learning agents
- Managing episode lifecycle

### Network

The network class (`Network`) provides a unified interface to the SUMO simulation. It handles:
- Lane-level traffic signal control
- Traffic metrics collection (waiting times, queue lengths, etc.)
- Network graph representation
- Export of network data for visualization

### Agents

Multiple agent types are implemented to control traffic signals:
- Base agent class provides the interface
- Specialized implementations use different machine learning approaches
- Agent factory pattern for creating appropriate agents

## Getting Started

### Prerequisites

- Python 3.7+
- SUMO (Simulation of Urban MObility)
- Required Python packages (see dependencies)

### Running a Simulation

1. Ensure SUMO is installed and configured
2. Run the main application:
```
python src/runner.py
```
3. Follow the interactive prompts to configure the simulation:
   - Select a network configuration
   - Choose agent types to test
   - Set the number of training episodes
   - Choose between GUI and headless mode

### Resuming Existing Runs

The system supports resuming previous experiment runs:
1. Run the main application
2. Select "Resume existing run"
3. Choose from the list of available runs
4. Configure additional episodes to run

## Data Analysis

Simulation results are stored in Parquet format in the data directory. Each run has:
- Step-level data: Detailed metrics for each simulation step
- Episode-level data: Summary metrics for each training episode

## Advanced Features

### Early Episode Termination

The system includes configurable early termination criteria to avoid running simulations that have reached undesirable states:
- Maximum waiting time threshold
- Maximum queue length threshold
- Minimum steps before checking termination conditions

## Development

### Adding New Agents

To implement a new agent type:
1. Create a new class in the `/src/agents` directory that inherits from `Agent`
2. Implement required methods: `choose_action`, `learn`, `calculate_reward`
3. Add the agent to the agent factory in `agent_factory.py`

### Adding New Networks

To add a new traffic network:
1. Create a SUMO configuration in the `/networks` directory
2. Add the network to the available networks list in `runner.py`


## Acknowledgments

- SUMO team for the traffic simulation platform
