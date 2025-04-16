# [V2-6.1] Pytest fixtures for LLM Manager v2
import pytest
from llm_manager_v2.app import create_app

@pytest.fixture(scope="module")
def test_client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
