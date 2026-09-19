import os
import pytest
from config.settings import get_settings
from src.utils.logger import get_logger
from src.monitoring.cost_tracker import calculate_cost, CostTrackerCallback


def test_settings_load_defaults(monkeypatch):
    """Requirement: App settings should load environment variables cleanly

    with defaults.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

    settings = get_settings()
    assert settings.OPENAI_API_KEY == "test-openai-key"
    assert settings.QDRANT_URL == "http://localhost:6333"
    assert settings.MODEL_NAME == "gpt-4o-mini"
    assert settings.EMBEDDING_MODEL == "text-embedding-3-small"


def test_logger_initialization():
    """Requirement: Logger should instantiate without error and log structured

    messages.
    """
    logger = get_logger("test_module")
    assert logger is not None
    logger.info("Structured log test initialized", status="success", step=1)


def test_calculate_cost_gpt4o_mini():
    """Requirement 3: Cost tracking must accurately calculate model usage

    costs.
    """
    # gpt-4o-mini rates: $0.150 per 1M input tokens, $0.600 per 1M output tokens
    cost = calculate_cost(
        model_name="gpt-4o-mini", prompt_tokens=1000, completion_tokens=500
    )
    # Expected: (1000 * 0.00000015) + (500 * 0.0000006) = 0.00015 + 0.0003 = 0.00045
    assert round(cost, 6) == 0.00045


def test_cost_tracker_callback():
    """Requirement 3: LangChain callback handler tracks cumulative tokens and

    total cost.
    """
    tracker = CostTrackerCallback(model_name="gpt-4o-mini")

    # Simulate first LLM call run
    tracker.on_llm_end_tokens(prompt_tokens=2000, completion_tokens=1000)
    assert tracker.total_tokens == 3000
    assert round(tracker.total_cost_usd, 5) == 0.0009