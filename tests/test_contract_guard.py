from ingest.contract_guard import guard_model_output, safe_json_value


def test_safe_json_value_repairs_fenced_payload() -> None:
    payload = "```json\n{\"k\": 1}\n```"
    parsed = safe_json_value(payload)
    assert parsed == {"k": 1}


def test_guard_model_output_returns_parse_failure() -> None:
    decision = guard_model_output("{ bad", lambda raw: safe_json_value(raw))
    assert decision.should_fallback
    assert not decision.parse_success
    assert decision.failure_reason == "parse_failure"


def test_guard_model_output_returns_validator_failure_reason() -> None:
    decision = guard_model_output(
        "{\"page\": 1}",
        lambda raw: safe_json_value(raw),
        validator=lambda payload: (False, "page_mismatch"),
    )
    assert decision.should_fallback
    assert decision.parse_success
    assert not decision.validation_success
    assert decision.failure_reason == "page_mismatch"
