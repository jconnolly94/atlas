# agents/agent_factory.py
from typing import Dict, Any, Type, Callable, List, Optional
from src.utils.conflict_detector import ConflictDetector

from .agent import Agent
from .q_agent import QAgent
from .dqn_agent import DQNAgent
from .advanced_agent import AdvancedAgent
from .enhanced_agent import EnhancedAgent
from .no_agent import NoAgent
from .revised_agent import RevisedAgent  # Import the new RevisedAgent


class AgentFactory:
    """Factory for creating agents with different configurations."""

    def __init__(self):
        """Initialize the agent factory."""
        self.agent_types = {
            "Q-Learning": QAgent,
            "DQN": DQNAgent,
            "Advanced": AdvancedAgent, 
            "Enhanced": EnhancedAgent,
            "Revised": RevisedAgent,  # Add the new agent type
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
            },
            "Revised": {
                "epsilon": 1.0,              # Start with 100% exploration
                "epsilon_decay": 0.9995,     # Slower decay
                "epsilon_min": 0.1,          # Higher minimum exploration
                "alpha": 0.001,              # Higher learning rate
                "debug_interval": 50         # Log debug info every 50 steps
            }
        }

    def create_agent(self, agent_type: str, tls_id: str, network, conflict_detector: Optional[ConflictDetector] = None, **kwargs) -> Agent:
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
        
        # Check if agent requires a conflict detector
        if agent_type in ["Advanced", "Enhanced", "Revised"]:  # Add Revised to the list
            if conflict_detector is None:
                raise ValueError(f"ConflictDetector is required for agent type '{agent_type}' but was not provided.")
            # Pass the conflict detector to the agent
            return agent_class.create(tls_id, network, conflict_detector=conflict_detector, **config)
        else:
            # For agents that don't need the conflict detector
            return agent_class.create(tls_id, network, **config)

    def create_agents_for_network(self, agent_type: str, network, conflict_detector: Optional[ConflictDetector] = None) -> Dict[str, Agent]:
        """Create agents for all traffic lights in a network.

        Args:
            agent_type: Type of agent to create
            network: Network object providing access to simulation data
            conflict_detector: ConflictDetector instance for agents that need it

        Returns:
            Dictionary mapping traffic light IDs to agent instances
            
        Raises:
            ValueError: If a required conflict_detector is not provided for agents that need it
        """
        agents = {}

        for tls_id in network.tls_ids:
            agents[tls_id] = self.create_agent(agent_type, tls_id, network, conflict_detector=conflict_detector)

        return agents


# Singleton instance for convenience
agent_factory = AgentFactory()


# Helper functions for agent creation
def create_q_agent(tls_id, network, **kwargs):
    return agent_factory.create_agent("Q-Learning", tls_id, network, **kwargs)


def create_dqn_agent(tls_id, network, **kwargs):
    return agent_factory.create_agent("DQN", tls_id, network, **kwargs)


def create_advanced_agent(tls_id, network, conflict_detector=None, **kwargs):
    return agent_factory.create_agent("Advanced", tls_id, network, conflict_detector=conflict_detector, **kwargs)


def create_enhanced_agent(tls_id, network, conflict_detector=None, **kwargs):
    return agent_factory.create_agent("Enhanced", tls_id, network, conflict_detector=conflict_detector, **kwargs)


def create_revised_agent(tls_id, network, conflict_detector=None, **kwargs):
    return agent_factory.create_agent("Revised", tls_id, network, conflict_detector=conflict_detector, **kwargs)


def no_agent(tls_id, network, **kwargs):
    return agent_factory.create_agent("Baseline", tls_id, network, **kwargs)