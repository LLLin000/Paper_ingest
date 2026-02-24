import os

from ingest import reading
from ingest import vision


def _clear_sf_env() -> None:
    for key in (
        "SILICONFLOW_API_KEY",
        "SF_API_KEY",
        "SILICONFLOW_TOKEN",
        "SILICONFLOW_ENDPOINT",
    ):
        os.environ.pop(key, None)


def test_reading_preflight_fails_fast_without_credentials() -> None:
    _clear_sf_env()
    ok, diag = reading.run_preflight_check("dummy-reading-model")
    assert not ok
    assert diag["error_type"] == "missing_api_key"
    assert "SILICONFLOW_API_KEY" in diag["message"]


def test_reading_preflight_rejects_invalid_endpoint() -> None:
    _clear_sf_env()
    os.environ["SILICONFLOW_API_KEY"] = "token"
    os.environ["SILICONFLOW_ENDPOINT"] = "http://localhost:9999/api"
    ok, diag = reading.run_preflight_check("dummy-reading-model")
    assert not ok
    assert diag["error_type"] == "invalid_endpoint"


def test_vision_preflight_rejects_invalid_endpoint() -> None:
    _clear_sf_env()
    os.environ["SILICONFLOW_API_KEY"] = "token"
    os.environ["SILICONFLOW_ENDPOINT"] = "https://api.siliconflow.cn/v1/models"
    ok, diag = vision.run_preflight_check("dummy-vision-model")
    assert not ok
    assert diag["error_type"] == "invalid_endpoint"


def test_vision_retry_policy_distinguishes_request_contract_from_transient() -> None:
    assert vision.is_request_contract_error("request_contract_error", 400)
    assert not vision.is_transient_retryable_error("request_contract_error", 400)
    assert vision.is_transient_retryable_error("timeout", None)
    assert vision.is_transient_retryable_error("network_error", None)
