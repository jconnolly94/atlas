# agents/agent_factory.py
from typing import Dict, Any, Type, Callable, List, Optional

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
        
        # Default configurations for specific agent types
        self.agent_configs = {
            "Q-Learning": {
                "epsilon": 0.8,              # High exploration rate
                "epsilon_decay": 0.998,      # Slower decay for more exploration
                "min_epsilon": 0.15         # Higher minimum exploration
            },
            "DQN": {
                "epsilon": 0.9,              # Very high exploration rate
                "epsilon_decay": 0.9999,     # Very slow decay for exploration
                "epsilon_min": 0.2           # Higher minimum for exploration
            }
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

        # Get agent class
        agent_class = self.agent_types[agent_type]
        
        # Apply default configuration if available
        config = kwargs.copy()
        if agent_type in self.agent_configs:
            # Start with default configs
            default_config = self.agent_configs[agent_type].copy()
            # Override defaults with any provided kwargs
            default_config.update(config)
            config = default_config
        
        # Initialize the agent
        return agent_class.create(tls_id, network, **config)

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


# Helper functions for agent creation
def create_q_agent(tls_id, network, **kwargs):
    return agent_factory.create_agent("Q-Learning", tls_id, network, **kwargs)


def create_dqn_agent(tls_id, network, **kwargs):
    return agent_factory.create_agent("DQN", tls_id, network, **kwargs)


def create_advanced_agent(tls_id, network, **kwargs):
    return agent_factory.create_agent("Advanced", tls_id, network, **kwargs)


def no_agent(tls_id, network, **kwargs):
    return agent_factory.create_agent("Baseline", tls_id, network, **kwargs)