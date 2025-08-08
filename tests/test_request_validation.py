import pytest
from fastapi import status


def test_temperature_out_of_range(api_client, sample_request_data):
	sample_request_data["temperature"] = 3.5
	resp = api_client.post("/v1/responses", json=sample_request_data)
	assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
	body = resp.json()
	# Ensure we have at least one validation error entry
	assert isinstance(body.get("detail"), list) and len(body["detail"]) > 0


def test_negative_max_output_tokens(api_client, sample_request_data):
	sample_request_data["max_output_tokens"] = -5
	resp = api_client.post("/v1/responses", json=sample_request_data)
	assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_reasoning_effort_legacy_field_low(api_client, sample_request_data):
	# remove nested reasoning to ensure legacy field is applied
	sample_request_data.pop("reasoning", None)
	sample_request_data["reasoning_effort"] = "low"
	resp = api_client.post("/v1/responses", json=sample_request_data)
	assert resp.status_code == status.HTTP_200_OK
	data = resp.json()
	assert data["model"] == sample_request_data["model"]


def test_reasoning_effort_legacy_field_medium(api_client, sample_request_data):
	sample_request_data.pop("reasoning", None)
	sample_request_data["reasoning_effort"] = "medium"
	resp = api_client.post("/v1/responses", json=sample_request_data)
	assert resp.status_code == status.HTTP_200_OK


def test_session_id_passthrough(api_client, sample_request_data):
	sample_request_data["session_id"] = "sess_123"
	resp = api_client.post("/v1/responses", json=sample_request_data)
	assert resp.status_code == status.HTTP_200_OK


def test_previous_response_id_alias(api_client, sample_request_data):
	# first create a response
	first = api_client.post("/v1/responses", json=sample_request_data)
	assert first.status_code == status.HTTP_200_OK
	rid = first.json()["id"]

	continuation = {
		"model": sample_request_data["model"],
		"response_id": rid,  # alias for previous_response_id
		"input": "Follow up via alias"
	}
	resp = api_client.post("/v1/responses", json=continuation)
	assert resp.status_code == status.HTTP_200_OK
