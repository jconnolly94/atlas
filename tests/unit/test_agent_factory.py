import unittest
from unittest.mock import MagicMock, patch
from src.agents.agent_factory import AgentFactory, agent_factory
from src.agents.agent import Agent
from src.agents.q_agent import QAgent
from src.agents.dqn_agent import DQNAgent
from src.agents.advanced_agent import AdvancedAgent
from src.agents.enhanced_agent import EnhancedAgent
from src.agents.no_agent import NoAgent


class TestAgentFactory(unittest.TestCase):
    """Test case for the AgentFactory class."""

    def setUp(self):
        """Set up test fixtures."""
        self.factory = AgentFactory()
        self.mock_network = MagicMock()

    def test_initialization(self):
        """Test AgentFactory initialization."""
        # Check agent types dictionary
        self.assertEqual(self.factory.agent_types["Q-Learning"], QAgent)
        self.assertEqual(self.factory.agent_types["DQN"], DQNAgent)
        self.assertEqual(self.factory.agent_types["Advanced"], AdvancedAgent)
        self.assertEqual(self.factory.agent_types["Enhanced"], EnhancedAgent)
        self.assertEqual(self.factory.agent_types["Baseline"], NoAgent)

    @patch('src.agents.q_agent.QAgent.create')
    def test_create_q_agent(self, mock_create):
        """Test creating a Q-Learning agent."""
        # Setup mock
        mock_agent = MagicMock()
        mock_create.return_value = mock_agent
        
        # Create agent
        agent = self.factory.create_agent("Q-Learning", "tls1", self.mock_network, alpha=0.2)
        
        # Verify agent was created
        mock_create.assert_called_once_with("tls1", self.mock_network, alpha=0.2)
        self.assertEqual(agent, mock_agent)

    @patch('src.agents.dqn_agent.DQNAgent.create')
    def test_create_dqn_agent(self, mock_create):
        """Test creating a DQN agent."""
        # Setup mock
        mock_agent = MagicMock()
        mock_create.return_value = mock_agent
        
        # Create agent
        agent = self.factory.create_agent("DQN", "tls1", self.mock_network, epsilon=0.5)
        
        # Verify agent was created
        mock_create.assert_called_once_with("tls1", self.mock_network, epsilon=0.5)
        self.assertEqual(agent, mock_agent)

    @patch('src.agents.advanced_agent.AdvancedAgent.create')
    def test_create_advanced_agent(self, mock_create):
        """Test creating an Advanced agent."""
        # Setup mock
        mock_agent = MagicMock()
        mock_create.return_value = mock_agent
        
        # Create agent
        agent = self.factory.create_agent("Advanced", "tls1", self.mock_network)
        
        # Verify agent was created
        mock_create.assert_called_once_with("tls1", self.mock_network)
        self.assertEqual(agent, mock_agent)

    @patch('src.agents.enhanced_agent.EnhancedAgent.create')
    def test_create_enhanced_agent(self, mock_create):
        """Test creating an Enhanced agent."""
        # Setup mock
        mock_agent = MagicMock()
        mock_create.return_value = mock_agent
        
        # Create agent
        agent = self.factory.create_agent("Enhanced", "tls1", self.mock_network, use_coordination=False)
        
        # Verify agent was created
        mock_create.assert_called_once_with("tls1", self.mock_network, use_coordination=False)
        self.assertEqual(agent, mock_agent)

    def test_create_unknown_agent(self):
        """Test creating an unknown agent type."""
        with self.assertRaises(ValueError):
            self.factory.create_agent("Unknown", "tls1", self.mock_network)

    @patch('src.agents.q_agent.QAgent.create')
    def test_create_agents_for_network(self, mock_create):
        """Test creating agents for all traffic lights in a network."""
        # Setup mock network
        self.mock_network.tls_ids = ["tls1", "tls2", "tls3"]
        
        # Setup mock agent creation
        mock_agents = [MagicMock() for _ in range(3)]
        mock_create.side_effect = mock_agents
        
        # Create agents for network
        agents = self.factory.create_agents_for_network("Q-Learning", self.mock_network)
        
        # Verify agents were created for each traffic light
        self.assertEqual(len(agents), 3)
        self.assertEqual(agents["tls1"], mock_agents[0])
        self.assertEqual(agents["tls2"], mock_agents[1])
        self.assertEqual(agents["tls3"], mock_agents[2])
        
        # Verify create was called for each traffic light
        self.assertEqual(mock_create.call_count, 3)
        mock_create.assert_any_call("tls1", self.mock_network)
        mock_create.assert_any_call("tls2", self.mock_network)
        mock_create.assert_any_call("tls3", self.mock_network)

    @patch('src.agents.agent_factory.agent_factory.create_agent')
    def test_helper_functions(self, mock_create_agent):
        """Test the helper functions for agent creation."""
        from src.agents.agent_factory import (
            create_q_agent, create_dqn_agent, create_advanced_agent,
            create_enhanced_agent, no_agent
        )
        
        # Setup mock
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        
        # Test all helper functions
        q_agent = create_q_agent("tls1", self.mock_network, alpha=0.2)
        mock_create_agent.assert_called_with("Q-Learning", "tls1", self.mock_network, alpha=0.2)
        
        dqn_agent = create_dqn_agent("tls1", self.mock_network, epsilon=0.5)
        mock_create_agent.assert_called_with("DQN", "tls1", self.mock_network, epsilon=0.5)
        
        adv_agent = create_advanced_agent("tls1", self.mock_network)
        mock_create_agent.assert_called_with("Advanced", "tls1", self.mock_network)
        
        enh_agent = create_enhanced_agent("tls1", self.mock_network)
        mock_create_agent.assert_called_with("Enhanced", "tls1", self.mock_network)
        
        baseline = no_agent("tls1", self.mock_network)
        mock_create_agent.assert_called_with("Baseline", "tls1", self.mock_network)
        
        # Verify all returned the mock agent
        self.assertEqual(q_agent, mock_agent)
        self.assertEqual(dqn_agent, mock_agent)
        self.assertEqual(adv_agent, mock_agent)
        self.assertEqual(enh_agent, mock_agent)
        self.assertEqual(baseline, mock_agent)
        
        # Verify create_agent was called the correct number of times
        self.assertEqual(mock_create_agent.call_count, 5)


if __name__ == '__main__':
    unittest.main()