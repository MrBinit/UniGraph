from app.scripts import eval_queue_worker, llm_async_worker, metrics_aggregation_worker


def _assert_main_validates_security(monkeypatch, module):
    calls: list[str] = []

    async def fake_run_forever():
        calls.append("run_forever")

    monkeypatch.setattr(module, "run_forever", fake_run_forever)
    monkeypatch.setattr(module, "validate_security_configuration", lambda: calls.append("validate"))
    monkeypatch.setattr(module.logging, "basicConfig", lambda **_kwargs: None)

    module.main()

    assert calls
    assert calls[0] == "validate"
    assert "run_forever" in calls


def test_llm_async_worker_main_validates_security(monkeypatch):
    _assert_main_validates_security(monkeypatch, llm_async_worker)


def test_eval_queue_worker_main_validates_security(monkeypatch):
    _assert_main_validates_security(monkeypatch, eval_queue_worker)


def test_metrics_worker_main_validates_security(monkeypatch):
    _assert_main_validates_security(monkeypatch, metrics_aggregation_worker)
