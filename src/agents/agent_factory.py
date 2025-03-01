# agents/agent_factory.py
from typing import Dict, Any, Type, Callable

from .agent import Agent
from .q_agent import QAgent
from .dqn_agent import DQNAgent
from .advanced_agent import AdvancedAgent
from .no_agent import NoAgent


class AgentFactory:
    """Factory for creating agents with different configurations."""

    def __init__(self):
        """Initialize the agent factory."""
        self.agent_types = {
            "Q-Learning": QAgent,
            "DQN": DQNAgent,
            "Advanced": AdvancedAgent,
            "Baseline": NoAgent
        }

        # Default configurations for different agent types
        self.default_configs = {
            "Q-Learning": {
                "alpha": 0.1,
                "gamma": 0.9,
                "epsilon": 0.1,
                "state_bin_size": 5
            },
            "DQN": {
                "alpha": 0.001,
                "gamma": 0.95,
                "epsilon": 0.1,
                "epsilon_decay": 0.995,
                "epsilon_min": 0.01,
                "batch_size": 32,
                "memory_size": 10000,
                "target_update_freq": 100
            },
            "Advanced": {
                "alpha": 0.001,
                "gamma": 0.95,
                "epsilon": 0.1,
                "epsilon_decay": 0.995,
                "epsilon_min": 0.01,
                "batch_size": 32,
                "memory_size": 10000,
                "target_update_freq": 100
            },
            "Baseline": {}
        }

    def create_agent(self, agent_type: str, tls_id: str, network, **kwargs) -> Agent:
        """Create an agent of the specified type.

        Args:
            agent_type: Type of agent to create
            tls_id: ID of the traffic light this agent controls
            network: Network object providing access to simulation data
            **kwargs: Additional configuration parameters

        Returns:
            An agent instance of the specified type

        Raises:
            ValueError: If the agent type is unknown
        """
        if agent_type not in self.agent_types:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Get agent class and default config
        agent_class = self.agent_types[agent_type]
        config = self.default_configs[agent_type].copy()

        # Override defaults with provided kwargs
        config.update(kwargs)

        # For the advanced agent, we need to provide additional parameters
        # related to its action space
        if agent_type == "Advanced":
            # Get possible phases for this traffic light
            possible_phases = network.get_possible_phases(tls_id)

            # Default duration options (in seconds)
            duration_options = [5, 10, 15, 20]

            # State size based on environment
            state_size = 6  # Default size based on state features

            # Create agent with combined parameters
            agent = agent_class(
                possible_phases=possible_phases,
                duration_options=duration_options,
                state_size=state_size,
                **config
            )

            # Special handling: set traffic light association
            agent.set_traffic_light(tls_id, network)

            return agent

        # For other agents, just create with standard parameters
        return agent_class(tls_id, network, **config)

    def create_agents_for_network(self, agent_type: str, network) -> Dict[str, Agent]:
        """Create agents for all traffic lights in a network.

        Args:
            agent_type: Type of agent to create
            network: Network object providing access to simulation data

        Returns:
            Dictionary mapping traffic light IDs to agent instances
        """
        agents = {}

        for tls_id in network.tls_ids:
            agents[tls_id] = self.create_agent(agent_type, tls_id, network)

        return agents


# Singleton instance for convenience
agent_factory = AgentFactory()


# Helper functions for backward compatibility
def create_q_agent(tls_id, network, **kwargs):
    return agent_factory.create_agent("Q-Learning", tls_id, network, **kwargs)


def create_dqn_agent(tls_id, network, **kwargs):
    return agent_factory.create_agent("DQN", tls_id, network, **kwargs)


def create_advanced_agent(tls_id, network, **kwargs):
    return agent_factory.create_agent("Advanced", tls_id, network, **kwargs)


def no_agent(tls_id, network, **kwargs):
    return agent_factory.create_agent("Baseline", tls_id, network, **kwargs)