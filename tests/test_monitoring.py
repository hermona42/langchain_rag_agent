import os
import pytest
from config.settings import get_settings
from src.utils.logger import get_logger

def test_settings_load_defaults(monkeypatch):
    """
    Requirement: App settings should load environment variables cleanly
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
    """
    Requirement: Logger should instantiate without error and log structured messages.
    """
    logger = get_logger("test_module")
    assert logger is not None
    # Verify execution does not throw
    logger.info("Structured log test initialized", status="success", step=1)