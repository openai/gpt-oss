from typing import Any, Dict, Literal, Optional, Union

from openai_harmony import ReasoningEffort
from pydantic import BaseModel

MODEL_IDENTIFIER = "gpt-oss-120b"
DEFAULT_TEMPERATURE = 0.0
REASONING_EFFORT = ReasoningEffort.LOW
DEFAULT_MAX_OUTPUT_TOKENS = 131072


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
    annotations: Optional[list[UrlCitation]] = []


class SummaryTextContentItem(BaseModel):
    # using summary for compatibility with the existing API
    type: Literal["summary_text"]
    text: str


class ReasoningTextContentItem(BaseModel):
    type: Literal["text"]  # Changed from reasoning_text to text
    text: str


class ReasoningItem(BaseModel):
    id: str = "rs_1234"
    type: Literal["reasoning"]
    summary: list[ReasoningTextContentItem]  # Use ReasoningTextContentItem for summary
    status: Optional[Literal["in_progress", "completed", "incomplete"]] = "completed"


class Item(BaseModel):
    type: Optional[Literal["message"]] = "message"
    role: Literal["user", "assistant", "system", "developer"]
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


class CodeInterpreterCallItem(BaseModel):
    type: Literal["code_interpreter_call"]
    id: str = "ci_1234"
    status: Literal["in_progress", "completed", "incomplete"] = "completed"
    input: Optional[str] = None


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


class CodeInterpreterToolConfig(BaseModel):
    type: Literal["code_interpreter"]


class MCPToolConfig(BaseModel):
    type: Literal["mcp"]
    server_label: str
    server_url: str
    server_description: Optional[str] = None
    headers: Optional[Dict[str, str]] = {}
    require_approval: Optional[Literal["always", "never"]] = "never"
    allowed_tools: Optional[list[str]] = []


class WebSearchPreviewToolConfig(BaseModel):
    type: Literal["web_search_preview"]


class TextFormatConfig(BaseModel):
    type: Literal["text"] = "text"


class TextConfig(BaseModel):
    format: Optional[TextFormatConfig] = TextFormatConfig()
    verbosity: Optional[Literal["low", "medium", "high"]] = "medium"


class ReasoningConfig(BaseModel):
    effort: Literal["low", "medium", "high"] = REASONING_EFFORT
    summary: Optional[Literal["detailed", "brief"]] = None


class ResponsesRequest(BaseModel):
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = DEFAULT_MAX_OUTPUT_TOKENS
    input: Union[
        str,
        list[
            Union[
                Item,
                ReasoningItem,
                FunctionCallItem,
                FunctionCallOutputItem,
                WebSearchCallItem,
            ]
        ],
    ]
    model: Optional[str] = MODEL_IDENTIFIER
    stream: Optional[bool] = False
    tools: Optional[
        list[
            Union[FunctionToolDefinition, BrowserToolConfig, CodeInterpreterToolConfig, MCPToolConfig, WebSearchPreviewToolConfig]
        ]
    ] = []
    reasoning: Optional[ReasoningConfig] = ReasoningConfig()
    metadata: Optional[Dict[str, Any]] = {}
    tool_choice: Optional[Literal["auto", "none"]] = "auto"
    parallel_tool_calls: Optional[bool] = False
    store: Optional[bool] = False
    previous_response_id: Optional[str] = None
    temperature: Optional[float] = DEFAULT_TEMPERATURE
    include: Optional[list[str]] = None
    text: Optional[TextConfig] = None


class ResponseObject(BaseModel):
    output: list[
        Union[
            Item,
            ReasoningItem,
            FunctionCallItem,
            FunctionCallOutputItem,
            WebSearchCallItem,
            CodeInterpreterCallItem,
        ]
    ]
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
