import pytest
import json
import asyncio
from fastapi import status
from unittest.mock import patch, MagicMock, AsyncMock


class TestResponsesEndpoint:
    
    def test_basic_response_creation(self, api_client, sample_request_data):
        response = api_client.post("/v1/responses", json=sample_request_data)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "id" in data
        assert data["object"] == "response"
        assert data["model"] == sample_request_data["model"]
    
    def test_response_with_high_reasoning(self, api_client, sample_request_data):
        sample_request_data["reasoning"] = {"effort": "high"}
        response = api_client.post("/v1/responses", json=sample_request_data)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "id" in data
        assert data["status"] == "completed"
        # Should have reasoning output
        assert len(data["output"]) >= 1
        assert any(item["type"] == "reasoning" for item in data["output"])
    
    def test_response_with_medium_reasoning(self, api_client, sample_request_data):
        sample_request_data["reasoning"] = {"effort": "medium"}
        response = api_client.post("/v1/responses", json=sample_request_data)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "id" in data
        assert data["status"] == "completed"
    
    def test_response_with_invalid_model(self, api_client, sample_request_data):
        sample_request_data["model"] = "invalid-model"
        response = api_client.post("/v1/responses", json=sample_request_data)
        # Should still accept but might handle differently
        assert response.status_code == status.HTTP_200_OK
    
    def test_response_with_empty_input(self, api_client, sample_request_data):
        sample_request_data["input"] = ""
        response = api_client.post("/v1/responses", json=sample_request_data)
        assert response.status_code == status.HTTP_200_OK
    
    def test_response_with_tools(self, api_client, sample_request_data):
        sample_request_data["tools"] = [
            {
                "type": "browser_search"
            }
        ]
        response = api_client.post("/v1/responses", json=sample_request_data)
        assert response.status_code == status.HTTP_200_OK
    
    def test_response_with_custom_temperature(self, api_client, sample_request_data):
        for temp in [0.0, 0.5, 1.0, 1.5, 2.0]:
            sample_request_data["temperature"] = temp
            response = api_client.post("/v1/responses", json=sample_request_data)
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "usage" in data
    
    def test_streaming_response(self, api_client, sample_request_data):
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
    
    def test_missing_required_fields(self, api_client):
        # Model field has default, so test with empty JSON
        response = api_client.post("/v1/responses", json={})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_invalid_reasoning_effort(self, api_client, sample_request_data):
        sample_request_data["reasoning"] = {"effort": "invalid"}
        response = api_client.post("/v1/responses", json=sample_request_data)
        # Should return validation error
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_malformed_json(self, api_client):
        response = api_client.post(
            "/v1/responses",
            data="not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_extremely_long_input(self, api_client, sample_request_data):
        # Test with very long input
        sample_request_data["input"] = "x" * 100000
        response = api_client.post("/v1/responses", json=sample_request_data)
        assert response.status_code == status.HTTP_200_OK


class TestToolIntegration:
    
    def test_browser_search_tool(self, api_client, sample_request_data):
        sample_request_data["tools"] = [
            {
                "type": "browser_search"
            }
        ]
        response = api_client.post("/v1/responses", json=sample_request_data)
        assert response.status_code == status.HTTP_200_OK
    
    def test_function_tool_integration(self, api_client, sample_request_data):
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
        performance_timer.start()
        response = api_client.post("/v1/responses", json=sample_request_data)
        elapsed = performance_timer.stop()
        
        assert response.status_code == status.HTTP_200_OK
        # Response should be reasonably fast for mock inference
        assert elapsed < 5.0  # 5 seconds threshold
    
    def test_multiple_sequential_requests(self, api_client, sample_request_data):
        # Test multiple requests work correctly
        for i in range(3):
            data = sample_request_data.copy()
            data["input"] = f"Request {i}"
            response = api_client.post("/v1/responses", json=data)
            assert response.status_code == status.HTTP_200_OK


class TestCompatibilityFeatures:
    """Test new compatibility features added for OpenAI format support"""
    
    def test_developer_role_mapping(self, api_client):
        """Test that developer role gets mapped to system role"""
        request_data = {
            "input": [
                {
                    "type": "message",
                    "role": "developer", 
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello!"}]
                }
            ],
            "reasoning": {"effort": "low"}
        }
        response = api_client.post("/v1/responses", json=request_data)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "output" in data
    
    def test_mcp_tools_support(self, api_client, sample_request_data):
        """Test that MCP tools are accepted without errors"""
        sample_request_data["tools"] = [
            {
                "type": "mcp",
                "server_label": "test-server",
                "server_url": "https://example.com/mcp",
                "server_description": "Test MCP server",
                "headers": {"Authorization": "Bearer test-token"},
                "require_approval": "always",
                "allowed_tools": ["test_tool"]
            }
        ]
        response = api_client.post("/v1/responses", json=sample_request_data)
        assert response.status_code == status.HTTP_200_OK
    
    def test_web_search_preview_tool(self, api_client, sample_request_data):
        """Test web_search_preview tool type"""
        sample_request_data["tools"] = [
            {
                "type": "web_search_preview"
            }
        ]
        response = api_client.post("/v1/responses", json=sample_request_data)
        assert response.status_code == status.HTTP_200_OK
    
    def test_reasoning_summary_field(self, api_client, sample_request_data):
        """Test reasoning config with summary field"""
        sample_request_data["reasoning"] = {
            "effort": "low",
            "summary": "detailed"
        }
        response = api_client.post("/v1/responses", json=sample_request_data)
        assert response.status_code == status.HTTP_200_OK
    
    def test_text_format_config(self, api_client, sample_request_data):
        """Test text format configuration"""
        sample_request_data["text"] = {
            "format": {"type": "text"},
            "verbosity": "low"
        }
        response = api_client.post("/v1/responses", json=sample_request_data)
        assert response.status_code == status.HTTP_200_OK
    
    def test_openai_response_format(self, api_client, sample_request_data):
        """Test that response matches OpenAI Responses API format"""
        response = api_client.post("/v1/responses", json=sample_request_data)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check top-level structure
        required_fields = ["id", "object", "created_at", "model", "status", "output", "usage"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        assert data["object"] == "response"
        assert data["status"] in ["completed", "in_progress", "failed", "incomplete"]
        
        # Check output structure
        assert isinstance(data["output"], list)
        for item in data["output"]:
            assert "type" in item
            if item["type"] == "reasoning":
                assert "summary" in item  # Should use summary, not content
                assert isinstance(item["summary"], list)
                for summary_item in item["summary"]:
                    assert summary_item["type"] == "text"  # Should be "text", not "reasoning_text"
            elif item["type"] == "message":
                assert "role" in item
                assert "content" in item
                assert isinstance(item["content"], list)
                for content_item in item["content"]:
                    assert content_item["type"] == "output_text"  # Should be "output_text"
        
        # Check usage structure
        usage = data["usage"]
        assert "input_tokens" in usage
        assert "output_tokens" in usage
        assert "total_tokens" in usage


class TestUsageTracking:
    
    def test_usage_object_structure(self, api_client, sample_request_data):
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
        # Short input
        response1 = api_client.post("/v1/responses", json=sample_request_data)
        usage1 = response1.json()["usage"]
        
        # Longer input
        sample_request_data["input"] = sample_request_data["input"] * 10
        response2 = api_client.post("/v1/responses", json=sample_request_data)
        usage2 = response2.json()["usage"]
        
        # Longer input should use more tokens
        assert usage2["input_tokens"] > usage1["input_tokens"]