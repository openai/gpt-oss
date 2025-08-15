import pytest
import json
import asyncio
from fastapi import status
from unittest.mock import patch, MagicMock, AsyncMock


class TestResponsesEndpoint:
    
    def test_basic_response_creation(self, api_client, sample_request_data):
        """Test that a basic response can be created successfully."""
        response = api_client.post("/v1/responses", json=sample_request_data)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "id" in data
        assert data["object"] == "response"
        assert data["model"] == sample_request_data["model"]
    
    def test_response_with_high_reasoning(self, api_client, sample_request_data):
        """Test response creation with high reasoning effort."""
        sample_request_data["reasoning_effort"] = "high"
        response = api_client.post("/v1/responses", json=sample_request_data)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "id" in data
        assert data["status"] == "completed"
    
    def test_response_with_medium_reasoning(self, api_client, sample_request_data):
        """Test response creation with medium reasoning effort."""
        sample_request_data["reasoning_effort"] = "medium"
        response = api_client.post("/v1/responses", json=sample_request_data)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "id" in data
        assert data["status"] == "completed"
    
    def test_response_with_invalid_model(self, api_client, sample_request_data):
        """Test response when an invalid model is specified."""
        sample_request_data["model"] = "invalid-model"
        response = api_client.post("/v1/responses", json=sample_request_data)
        # Should still accept but might handle differently
        assert response.status_code == status.HTTP_200_OK
    
    def test_response_with_empty_input(self, api_client, sample_request_data):
        """Test response when input is an empty string."""
        sample_request_data["input"] = ""
        response = api_client.post("/v1/responses", json=sample_request_data)
        assert response.status_code == status.HTTP_200_OK
    
    def test_response_with_tools(self, api_client, sample_request_data):
        """Test response when tools are specified in the request."""
        sample_request_data["tools"] = [
            {
                "type": "browser_search"
            }
        ]
        response = api_client.post("/v1/responses", json=sample_request_data)
        assert response.status_code == status.HTTP_200_OK
    
    def test_response_with_custom_temperature(self, api_client, sample_request_data):
        """Test response with various temperature values."""
        for temp in [0.0, 0.5, 1.0, 1.5, 2.0]:
            sample_request_data["temperature"] = temp
            response = api_client.post("/v1/responses", json=sample_request_data)
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "usage" in data
    
    def test_streaming_response(self, api_client, sample_request_data):
        """Test streaming response functionality using SSE."""
        sample_request_data["stream"] = True
        with api_client.stream("POST", "/v1/responses", json=sample_request_data) as response:
            assert response.status_code == status.HTTP_200_OK
            # Verify we get SSE events
            for line in response.iter_lines():
                if line and line.startswith("data: "):
                    event_data = line[6:]  # Remove "data: " prefix
                    if event_data != "[DONE]":
                        json.loads(event_data)  # Should be valid JSON
                        break


class TestResponsesWithSession:
    
    def test_response_with_session_id(self, api_client, sample_request_data):
        """Test that responses with the same session_id are handled correctly."""
        session_id = "test-session-123"
        sample_request_data["session_id"] = session_id
        
        # First request
        response1 = api_client.post("/v1/responses", json=sample_request_data)
        assert response1.status_code == status.HTTP_200_OK
        data1 = response1.json()
        
        # Second request with same session
        sample_request_data["input"] = "Follow up question"
        response2 = api_client.post("/v1/responses", json=sample_request_data)
        assert response2.status_code == status.HTTP_200_OK
        data2 = response2.json()
        
        # Should have different response IDs
        assert data1["id"] != data2["id"]
    
    def test_response_continuation(self, api_client, sample_request_data):
        """Test continuation of a previous response using response_id."""
        # Create initial response
        response1 = api_client.post("/v1/responses", json=sample_request_data)
        assert response1.status_code == status.HTTP_200_OK
        data1 = response1.json()
        response_id = data1["id"]
        
        # Continue the response
        continuation_request = {
            "model": sample_request_data["model"],
            "response_id": response_id,
            "input": "Continue the previous thought"
        }
        response2 = api_client.post("/v1/responses", json=continuation_request)
        assert response2.status_code == status.HTTP_200_OK


class TestErrorHandling:
    def test_invalid_types_in_payload(self, api_client, sample_request_data):
        # Set input to an integer instead of a string
        sample_request_data["input"] = 12345
        response = api_client.post("/v1/responses", json=sample_request_data)
        # Should return 422 Unprocessable Entity or 400 Bad Request
        assert response.status_code in [status.HTTP_422_UNPROCESSABLE_ENTITY, status.HTTP_400_BAD_REQUEST]

        # Set model to a list instead of a string
        sample_request_data["model"] = ["gpt-oss-20b"]
        response = api_client.post("/v1/responses", json=sample_request_data)
        assert response.status_code in [status.HTTP_422_UNPROCESSABLE_ENTITY, status.HTTP_400_BAD_REQUEST]

        # Set tools to a string instead of a list
        sample_request_data["tools"] = "browser_search"
        response = api_client.post("/v1/responses", json=sample_request_data)
        assert response.status_code in [status.HTTP_422_UNPROCESSABLE_ENTITY, status.HTTP_400_BAD_REQUEST]
    
    def test_missing_required_fields(self, api_client):
        """Test response when required fields are missing from the request."""
        # Model field has default, so test with empty JSON
        response = api_client.post("/v1/responses", json={})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_invalid_reasoning_effort(self, api_client, sample_request_data):
        """Test response when reasoning_effort has an invalid value."""
        sample_request_data["reasoning_effort"] = "invalid"
        response = api_client.post("/v1/responses", json=sample_request_data)
        # May handle gracefully or return error
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_422_UNPROCESSABLE_ENTITY]
    
    def test_malformed_json(self, api_client):
        """Test response when the request body is not valid JSON."""
        response = api_client.post(
            "/v1/responses",
            data="not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_extremely_long_input(self, api_client, sample_request_data):
        """Test response when input is extremely long."""
        # Test with very long input
        sample_request_data["input"] = "x" * 100000
        response = api_client.post("/v1/responses", json=sample_request_data)
        assert response.status_code == status.HTTP_200_OK


class TestToolIntegration:
    
    def test_browser_search_tool(self, api_client, sample_request_data):
        """Test response when browser_search tool is specified."""
        sample_request_data["tools"] = [
            {
                "type": "browser_search"
            }
        ]
        response = api_client.post("/v1/responses", json=sample_request_data)
        assert response.status_code == status.HTTP_200_OK
    
    def test_function_tool_integration(self, api_client, sample_request_data):
        """Test response when a function tool is specified."""
        sample_request_data["tools"] = [
            {
                "type": "function",
                "name": "test_function",
                "parameters": {"type": "object", "properties": {}},
                "description": "Test function"
            }
        ]
        response = api_client.post("/v1/responses", json=sample_request_data)
        assert response.status_code == status.HTTP_200_OK
    
    def test_multiple_tools(self, api_client, sample_request_data):
        """Test response when multiple tools are specified."""
        sample_request_data["tools"] = [
            {
                "type": "browser_search"
            },
            {
                "type": "function",
                "name": "test_function",
                "parameters": {"type": "object", "properties": {}},
                "description": "Test function"
            }
        ]
        response = api_client.post("/v1/responses", json=sample_request_data)
        assert response.status_code == status.HTTP_200_OK


class TestPerformance:
    
    def test_response_time_under_threshold(self, api_client, sample_request_data, performance_timer):
        """Test that response time is under a specified threshold."""
        performance_timer.start()
        response = api_client.post("/v1/responses", json=sample_request_data)
        elapsed = performance_timer.stop()
        
        assert response.status_code == status.HTTP_200_OK
        # Response should be reasonably fast for mock inference
        assert elapsed < 5.0  # 5 seconds threshold
    
    def test_multiple_sequential_requests(self, api_client, sample_request_data):
        """Test that multiple sequential requests are handled correctly."""
        # Test multiple requests work correctly
        for i in range(3):
            data = sample_request_data.copy()
            data["input"] = f"Request {i}"
            response = api_client.post("/v1/responses", json=data)
            assert response.status_code == status.HTTP_200_OK


class TestUsageTracking:
    
    def test_usage_object_structure(self, api_client, sample_request_data):
        """Test that the usage object in the response has the correct structure and values."""
        response = api_client.post("/v1/responses", json=sample_request_data)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "usage" in data
        usage = data["usage"]
        assert "input_tokens" in usage
        assert "output_tokens" in usage
        assert "total_tokens" in usage
        # reasoning_tokens may not always be present
        # assert "reasoning_tokens" in usage
        
        # Basic validation
        assert usage["input_tokens"] >= 0
        assert usage["output_tokens"] >= 0
        assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    
    def test_usage_increases_with_longer_input(self, api_client, sample_request_data):
        """Test that input_tokens increases as the input length increases."""
        # Short input
        response1 = api_client.post("/v1/responses", json=sample_request_data)
        usage1 = response1.json()["usage"]
        
        # Longer input
        sample_request_data["input"] = sample_request_data["input"] * 10
        response2 = api_client.post("/v1/responses", json=sample_request_data)
        usage2 = response2.json()["usage"]
        
        # Longer input should use more tokens
        assert usage2["input_tokens"] > usage1["input_tokens"]