"""Tests for drift_runtime.data.fetch — mocked HTTP."""

from unittest.mock import patch, MagicMock
import pytest
from drift_runtime.data import fetch
from drift_runtime.exceptions import DriftNetworkError
import httpx


def _mock_response(status_code=200, json_data=None, text="", content_type="application/json"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {"content-type": content_type}
    resp.json.return_value = json_data
    resp.text = text
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"{status_code}", request=MagicMock(), response=resp
        )
    return resp


def test_fetch_json():
    mock_resp = _mock_response(json_data=[{"id": 1}])
    with patch("drift_runtime.data.httpx.get", return_value=mock_resp):
        result = fetch("https://api.example.com/data")
    assert result == [{"id": 1}]


def test_fetch_with_headers():
    mock_resp = _mock_response(json_data={"ok": True})
    with patch("drift_runtime.data.httpx.get", return_value=mock_resp) as mock_get:
        fetch("https://api.example.com", headers={"X-Key": "abc"})
    mock_get.assert_called_once_with(
        "https://api.example.com",
        headers={"X-Key": "abc"},
        params=None,
        timeout=30,
    )


def test_fetch_with_params():
    mock_resp = _mock_response(json_data=[])
    with patch("drift_runtime.data.httpx.get", return_value=mock_resp) as mock_get:
        fetch("https://api.example.com", params={"limit": 50})
    mock_get.assert_called_once_with(
        "https://api.example.com",
        headers=None,
        params={"limit": 50},
        timeout=30,
    )


def test_fetch_non_json():
    mock_resp = _mock_response(text="hello world", content_type="text/plain")
    with patch("drift_runtime.data.httpx.get", return_value=mock_resp):
        result = fetch("https://example.com/text")
    assert result == "hello world"


def test_fetch_http_error_raises_drift_network_error():
    mock_resp = _mock_response(status_code=404)
    with patch("drift_runtime.data.httpx.get", return_value=mock_resp):
        with pytest.raises(DriftNetworkError, match="HTTP 404"):
            fetch("https://api.example.com/missing")


def test_fetch_connection_error_raises_drift_network_error():
    with patch("drift_runtime.data.httpx.get", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(DriftNetworkError, match="Request failed"):
            fetch("https://unreachable.example.com")
