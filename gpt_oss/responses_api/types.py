from typing import Any, Dict, Literal, Optional, Union

from openai_harmony import ReasoningEffort
from pydantic import BaseModel

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


class ReasoningConfig(BaseModel):
    effort: Literal["low", "medium", "high"] = REASONING_EFFORT


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
            Union[FunctionToolDefinition, BrowserToolConfig, CodeInterpreterToolConfig]
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
