"""
[V2-6.1] Service logic tests for LLM Manager v2
"""
from llm_manager_v2.services.huggingface_service import search_models, get_trending_models
from llm_manager_v2.services.model_service import list_models

def test_search_models():
    results = search_models("llama")
    assert isinstance(results, list)
    assert any("llama" in m["name"] for m in results)

def test_get_trending_models():
    results = get_trending_models()
    assert isinstance(results, list)
    assert len(results) > 0

def test_list_models():
    models, status = list_models()
    assert isinstance(models, list)
    assert status == 200 or status == 500
