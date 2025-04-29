# ATLAS – Adaptive Traffic Light Automation System

A framework for experimenting with reinforcement learning agents for efficient traffic signal control.

## Overview

Atlas is a traffic signal control system that uses various reinforcement learning agents to optimize traffic flow. The system integrates with the SUMO (Simulation of Urban MObility) traffic simulator to provide a realistic environment for training and evaluating traffic control strategies.

## Features

- Multiple reinforcement learning agents:
  - Q-Learning Agent
  - Deep Q-Network (DQN) Agent
  - Advanced Agent with adaptive phase duration
  - Enhanced Agent with temporal considerations and coordination
  - Baseline (no learning) Agent for comparison
- Interactive and headless simulation modes
- Network visualization tools
- Data collection and performance metrics
- Parallel agent training
- Support for multiple traffic network configurations

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/atlas.git
   cd atlas
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install SUMO traffic simulator:
   - Follow instructions at [SUMO Installation Guide](https://sumo.dlr.de/docs/Installing/index.html)
   - Make sure the SUMO_HOME environment variable is set correctly

## Usage

### Running the Simulation

Run in interactive mode:
```
python src/runner.py
```

Run in headless mode:
```
python src/runner.py --no-gui
```

### Available Networks

- Simple4wayCross: A basic 4-way intersection
- DublinRd: A model of Dublin Road network

### Agent Types

- **Q-Learning**: Traditional Q-learning with discrete state-action space
- **DQN**: Deep Q-Network using neural networks for function approximation
- **Advanced**: DQN with adaptive phase durations
- **Enhanced**: Advanced agent with temporal considerations and coordination between adjacent traffic lights
- **Baseline**: Fixed-time control strategy for comparison

## Project Structure

```
atlas/
├── networks/                # SUMO network configurations
│   ├── DublinRd/
│   └── Simple4wayCross/
├── src/
│   ├── agents/              # Agent implementations
│   │   ├── agent.py         # Base agent class
│   │   ├── q_agent.py       # Q-learning agent
│   │   ├── dqn_agent.py     # DQN agent
│   │   ├── advanced_agent.py # Advanced agent
│   │   ├── enhanced_agent.py # Enhanced agent
│   │   └── no_agent.py      # Baseline agent
│   ├── env/                 # Environment interfaces
│   │   ├── environment.py   # Main environment class
│   │   └── network.py       # Network interface for SUMO
│   ├── utils/               # Utility functions
│   │   ├── data_collector.py # Data collection tools
│   │   ├── network_exporter.py # Network visualization tools
│   │   └── observer.py      # Observer pattern implementation
│   └── runner.py            # Main entry point
├── tests/                   # Test suite
│   ├── agents/              # Agent unit tests
│   ├── env/                 # Environment unit tests
│   ├── utils/               # Utility unit tests
│   ├── integration/         # Integration tests
│   ├── run_tests.py         # Unit test runner
│   └── run_integration_tests.py # Integration test runner
```

## Learning Parameters

The agents use the following key learning parameters:

- **Learning Rate (α)**: Controls how quickly the agent updates its knowledge
- **Discount Factor (γ)**: Determines the importance of future rewards
- **Exploration Rate (ε)**: Controls the exploration-exploitation trade-off
- **ε-decay**: Gradually reduces exploration over time

These parameters are configured in `src/agents/agent_factory.py` and can be adjusted for different learning scenarios.

## Data Collection

Performance data is stored in:
- `data/datastore/steps/`: Step-by-step metrics
- `data/datastore/episodes/`: Episode-level metrics
- `data/datastore/logs/`: Log files for debugging

## Network Visualization

To export network data for visualization:
```
python src/utils/network_exporter.py
```

This generates network visualization files in `data/visualization/`.

## Testing

The project includes comprehensive test suites to ensure reliability and maintainability.

### Running Tests

Run unit tests:
```
python tests/run_tests.py
```

Run integration tests:
```
python tests/run_integration_tests.py
```

### Testing Strategy

- **Unit Tests**: Test individual components in isolation
  - Mock external dependencies (e.g., SUMO, traci)
  - Test class interfaces and behavior
  - Focus on core ML functionality rather than specific numeric outputs

- **Integration Tests**: Test interactions between components
  - Agent-environment interactions
  - Data collection during simulation
  - Network-SUMO interactions

Each test file follows a consistent pattern:
- Setup with mock objects
- Test initialization
- Test core functionality
- Test edge cases
- Proper teardown

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

When contributing:
1. Add tests for new functionality
2. Ensure all tests pass (`python tests/run_tests.py`)
3. Follow code style guidelines
4. Update documentation as needed

## Acknowledgements

- [SUMO Traffic Simulator](https://www.eclipse.org/sumo/)
- [PyTorch](https://pytorch.org/)
- [NetworkX](https://networkx.org/)