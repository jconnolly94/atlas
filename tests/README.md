# Tests for Atlas Traffic Control System

This directory contains tests for the Atlas Traffic Control System components.

## Test Organization

The tests are organized by type and module:

- **Unit Tests**:
  - `agents/`: Tests for agent implementations (Agent, QAgent, DQNAgent, etc.)
  - `env/`: Tests for environment and network interfaces
  - `utils/`: Tests for utility classes (DataCollector, MetricsCalculator, etc.)

- **Integration Tests**:
  - `integration/`: Tests for component interactions and system behavior

## Running Tests

### Unit Tests

Run all unit tests using the `run_tests.py` script:

```bash
python tests/run_tests.py
```

To run a specific test file or test case:

```bash
# Run a specific test file
python -m unittest tests.agents.test_agent

# Run a specific test case
python -m unittest tests.agents.test_agent.TestAgent

# Run a specific test method
python -m unittest tests.agents.test_agent.TestAgent.test_initialization
```

### Integration Tests

Run all integration tests using the `run_integration_tests.py` script:

```bash
python tests/run_integration_tests.py
```

To run a specific integration test file:

```bash
python -m unittest tests.integration.test_agent_environment
```

## Test Coverage

To run tests with coverage analysis:

```bash
# Install coverage package if not already installed
pip install coverage

# Run tests with coverage
coverage run --source=src tests/run_tests.py

# Include integration tests
coverage run --append --source=src tests/run_integration_tests.py

# View coverage report
coverage report

# Generate HTML coverage report
coverage html
```

## Test Guidelines

### Unit Tests

Unit tests should:
1. Test a single class or function in isolation
2. Mock all external dependencies
3. Cover both success and failure cases
4. Verify specific behaviors, not implementation details
5. Use real objects instead of mocks when testing functionality that's hard to mock (e.g., PyTorch models)
6. For ML components, test core functionality rather than specific numeric outputs
7. Include proper error handling for external system failures (e.g., SUMO connectivity issues)

### Integration Tests

Integration tests should:
1. Test interactions between two or more components
2. Focus on interface contracts between components
3. Use minimal mocking (only for external dependencies)
4. Verify system behavior when components are combined
5. Test error handling and recovery across component boundaries
6. Handle tensor operations carefully in PyTorch-related integration tests

## Key Integration Points

1. **Agent + Environment**: How agents interact with the environment, receive states, choose actions, and learn from rewards.

2. **Environment + Network**: How the environment uses the network interface to control and observe the simulation.

3. **Network + SUMO**: How the network class interacts with the SUMO traffic simulator via TraCI.

4. **Data Collection**: How the observer pattern is used to collect and record simulation data.

## Adding New Tests

When adding new components to the system, follow these guidelines:

1. Create a new test file in the appropriate subdirectory
2. Name the file `test_[module_name].py`
3. Create a test class for each class being tested
4. Name the test class `Test[ClassName]`
5. Include test methods for each function/method to test
6. Name test methods `test_[function_name]` or `test_[description]`
7. Use descriptive docstrings to document what each test is verifying

## Mock Objects

Use mock objects from `unittest.mock` to isolate the component being tested.
For example, when testing an agent, create a mock network rather than using
a real network instance:

```python
from unittest.mock import MagicMock

# Create a mock network
mock_network = MagicMock()
mock_network.get_controlled_lanes.return_value = ['lane1', 'lane2']

# Use the mock in your tests
agent = QAgent('tls1', mock_network)
```

### When to Use Real Objects vs. Mocks

While mocking is generally preferred for isolation, some components are better tested with simplified real implementations:

```python
# For PyTorch model testing, use a simple real model instead of mocks
class SimpleTestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(input_size, output_size, bias=False)
        # Initialize with predictable weights for testing
        torch.nn.init.ones_(self.layer.weight)
        
    def forward(self, x):
        return self.layer(x)

# Use in tests
model = SimpleTestModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

### Testing with Exception Handling

When testing components that might encounter errors, implement proper exception handling:

```python
# Testing error recovery
def test_network_export_error_handling(self):
    # Setup mock to cause error
    mock_exporter = MagicMock()
    mock_exporter.export_network_data.return_value = None  # Indicates failure
    
    try:
        # Should handle the error gracefully
        network = Network(self.config_file)
        network.export_network_data()
        # Verify network is still in a valid state
        self.assertIsNotNone(network.graph)
    except Exception as e:
        self.fail(f"export_network_data didn't handle error: {e}")
```

## Assertions

Use the appropriate assertion methods:

- `assertEqual`: Check if two values are equal
- `assertTrue`/`assertFalse`: Check boolean conditions
- `assertIn`: Check if a value is in a collection
- `assertIsInstance`: Check if an object is an instance of a class
- `assertRaises`: Check if a function raises an exception
- `assertAlmostEqual`: Check floating-point equality

## Common Testing Patterns

### Testing State Changes

```python
# Testing that a method changes object state
def test_epsilon_decay(self):
    initial_epsilon = self.agent.epsilon
    self.agent._replay()  # Method that should decay epsilon
    self.assertLess(self.agent.epsilon, initial_epsilon)
```

### Testing Method Calls

```python
# Testing that a method calls dependencies correctly
def test_build_from_sumo(self):
    network = Network(self.config_file)
    network.build_from_sumo()
    
    # Verify proper calls were made
    self.mock_traci.junction.getIDList.assert_called_once()
    self.mock_traci.edge.getIDList.assert_called_once()
```

### Skip Tests When Needed

```python
# Skip tests that need refinement
def test_complex_feature(self):
    self.skipTest("This test needs further refinement")
```