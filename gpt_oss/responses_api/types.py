from typing import Any, Dict, Literal, Optional, Union

from openai_harmony import ReasoningEffort
from pydantic import BaseModel, Field, field_validator, model_validator

MODEL_IDENTIFIER = "gpt-oss-120b"
DEFAULT_TEMPERATURE = 0.0
REASONING_EFFORT = ReasoningEffort.LOW
DEFAULT_MAX_OUTPUT_TOKENS = 10_000

class UrlCitation(BaseModel):
    type: Literal["url_citation"]
    end_index: int
    start_index: int
    url: str
    title: str

class TextContentItem(BaseModel):
    type: Union[Literal["text"], Literal["input_text"], Literal["output_text"]]
    text: str
    status: Optional[str] = "completed"
    annotations: Optional[list[UrlCitation]] = None


class SummaryTextContentItem(BaseModel):
    # using summary for compatibility with the existing API
    type: Literal["summary_text"]
    text: str


class ReasoningTextContentItem(BaseModel):
    type: Literal["reasoning_text"]
    text: str


class ReasoningItem(BaseModel):
    id: str = "rs_1234"
    type: Literal["reasoning"]
    summary: list[SummaryTextContentItem]
    content: Optional[list[ReasoningTextContentItem]] = []


class Item(BaseModel):
    type: Optional[Literal["message"]] = "message"
    role: Literal["user", "assistant", "system"]
    content: Union[list[TextContentItem], str]
    status: Union[Literal["in_progress", "completed", "incomplete"], None] = None


class FunctionCallItem(BaseModel):
    type: Literal["function_call"]
    name: str
    arguments: str
    status: Literal["in_progress", "completed", "incomplete"] = "completed"
    id: str = "fc_1234"
    call_id: str = "call_1234"


class FunctionCallOutputItem(BaseModel):
    type: Literal["function_call_output"]
    call_id: str = "call_1234"
    output: str

class WebSearchActionSearch(BaseModel):
    type: Literal["search"]
    query: Optional[str] = None

class WebSearchActionOpenPage(BaseModel):
    type: Literal["open_page"]
    url: Optional[str] = None

class WebSearchActionFind(BaseModel):
    type: Literal["find"]
    pattern: Optional[str] = None
    url: Optional[str] = None

class WebSearchCallItem(BaseModel):
    type: Literal["web_search_call"]
    id: str = "ws_1234"
    status: Literal["in_progress", "completed", "incomplete"] = "completed"
    action: Union[WebSearchActionSearch, WebSearchActionOpenPage, WebSearchActionFind]

class Error(BaseModel):
    code: str
    message: str


class IncompleteDetails(BaseModel):
    reason: str


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class FunctionToolDefinition(BaseModel):
    type: Literal["function"]
    name: str
    parameters: dict  # this should be typed stricter if you add strict mode
    strict: bool = False  # change this if you support strict mode
    description: Optional[str] = ""


class BrowserToolConfig(BaseModel):
    type: Literal["browser_search"]


class ReasoningConfig(BaseModel):
    effort: Literal["low", "medium", "high"] = REASONING_EFFORT


class ResponsesRequest(BaseModel):
    """Primary request schema for /v1/responses

    Backwards compatibility notes:
    - Accepts legacy field `reasoning_effort` (string) and maps it to `reasoning.effort`.
    - Accepts alias `response_id` for `previous_response_id` to match some client patterns.
    - Allows optional `session_id` (currently unused but often expected by clients).
    """
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = DEFAULT_MAX_OUTPUT_TOKENS
    input: Union[
        str, list[Union[Item, ReasoningItem, FunctionCallItem, FunctionCallOutputItem, WebSearchCallItem]]
    ]
    model: Optional[str] = MODEL_IDENTIFIER
    stream: Optional[bool] = False
    tools: Optional[list[Union[FunctionToolDefinition, BrowserToolConfig]]] = []
    reasoning: Optional[ReasoningConfig] = ReasoningConfig()
    # legacy single field support – user may pass reasoning_effort instead of nested structure
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None
    metadata: Optional[Dict[str, Any]] = {}
    tool_choice: Optional[Literal["auto", "none"]] = "auto"
    parallel_tool_calls: Optional[bool] = False
    store: Optional[bool] = False
    # Accept both previous_response_id and response_id (alias)
    previous_response_id: Optional[str] = Field(
        default=None, alias="response_id", description="Alias: response_id"
    )
    temperature: Optional[float] = DEFAULT_TEMPERATURE
    include: Optional[list[str]] = None
    # Optional session identifier (not yet used for server-side state but accepted)
    session_id: Optional[str] = None

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, v):  # noqa: D401
        """Validate temperature is within a reasonable range (0-2 inclusive)."""
        if v is None:
            return v
        if not (0.0 <= v <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0 inclusive")
        return v

    @field_validator("max_output_tokens")
    @classmethod
    def _validate_max_output_tokens(cls, v):
        if v is None:
            return v
        if v <= 0:
            raise ValueError("max_output_tokens must be > 0")
        return v

    @model_validator(mode="after")
    def _apply_reasoning_effort_legacy(self):
        # If user provided standalone reasoning_effort, map it into reasoning.effort
        if self.reasoning_effort is not None:
            if self.reasoning is None:
                self.reasoning = ReasoningConfig(effort=self.reasoning_effort)
            else:
                self.reasoning.effort = self.reasoning_effort  # type: ignore
        return self


class ResponseObject(BaseModel):
    output: list[Union[Item, ReasoningItem, FunctionCallItem, FunctionCallOutputItem, WebSearchCallItem]]
    created_at: int
    usage: Optional[Usage] = None
    status: Literal["completed", "failed", "incomplete", "in_progress"] = "in_progress"
    background: None = None
    error: Optional[Error] = None
    incomplete_details: Optional[IncompleteDetails] = None
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = None
    max_tool_calls: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = {}
    model: Optional[str] = MODEL_IDENTIFIER
    parallel_tool_calls: Optional[bool] = False
    previous_response_id: Optional[str] = None
    id: Optional[str] = "resp_1234"
    object: Optional[str] = "response"
    text: Optional[Dict[str, Any]] = None
    tool_choice: Optional[str] = "auto"
    top_p: Optional[int] = 1
