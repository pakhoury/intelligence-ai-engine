"""
Shared pytest fixtures for unit and integration tests.
"""
import sys
import os
from unittest.mock import MagicMock, AsyncMock, patch
import pytest

# Add src/rag-system to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "rag-system"))


@pytest.fixture
def mock_llm():
    """Mock LangChain LLM that returns configurable responses."""
    llm = MagicMock()
    llm.invoke = MagicMock()
    llm.ainvoke = AsyncMock()
    return llm


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    client = MagicMock()
    client.get = MagicMock(return_value=None)
    client.set = MagicMock(return_value=True)
    client.ping = MagicMock(return_value=True)
    return client


@pytest.fixture
def sample_state():
    """Base workflow state for testing."""
    return {
        "question": "How many compliance violations per severity level?",
        "session_id": "test-session",
        "conversation_history": [],
        "cache_hit": False,
        "route": "sql",
    }


@pytest.fixture
def sample_state_with_history():
    """Workflow state with conversation history."""
    return {
        "question": "Tell me more about the critical ones",
        "session_id": "test-session",
        "conversation_history": [
            {
                "question": "How many compliance violations per severity level?",
                "answer": "There are 2 Critical, 3 High, 2 Low, and 3 Medium violations.",
            }
        ],
        "cache_hit": False,
    }


@pytest.fixture
def sample_state_with_rich_history():
    """Workflow state with enriched conversation history (metric context + data)."""
    return {
        "question": "Break that down by department",
        "session_id": "test-session",
        "conversation_history": [
            {
                "question": "What is the compliance effectiveness score for Legal?",
                "answer": "The Compliance Effectiveness Score for the Legal department is 55.0%.",
                "resolved_metric": "compliance_effectiveness_score",
                "metric_version": "1.0",
                "route": "sql_only",
                "data_summary": "[('Legal', 100, 55, 55.0)]",
            }
        ],
        "cache_hit": False,
    }
