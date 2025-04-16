"""
[V2-6.1] Model/database smoke tests for LLM Manager v2
"""
from models.search_history import add_search_query, get_recent_searches, clear_search_history
from models.database import init_db, add_model, get_all_models, delete_model, reset_database

def test_search_history_crud():
    clear_search_history()
    add_search_query("test-query", "test-type")
    results = get_recent_searches()
    assert any(s["query"] == "test-query" for s in results)
    clear_search_history()
    assert len(get_recent_searches()) == 0

def test_database_crud(tmp_path):
    reset_database()
    name = "Test Model"
    framework = "test-framework"
    path = str(tmp_path / "test-model.bin")
    # Create a dummy file
    with open(path, "wb") as f:
        f.write(b"abc")
    result, status = add_model(name, framework, path)
    assert status == 201
    models, code = get_all_models()
    assert any(m["name"] == name for m in models)
    model_id = models[0]["id"]
    del_result, del_status = delete_model(model_id)
    assert del_status == 200
