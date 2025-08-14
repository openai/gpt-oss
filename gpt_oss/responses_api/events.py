# torchrun --nproc-per-node=4 responses_api.py
from typing import Literal, Optional, Union

from pydantic import BaseModel

from .types import (
    CodeInterpreterCallItem,
    FunctionCallItem,
    Item,
    ReasoningItem,
    ReasoningTextContentItem,
    ResponseObject,
    TextContentItem,
    UrlCitation,
    WebSearchCallItem,
)


class ResponseEvent(BaseModel):
    sequence_number: Optional[int] = 1


class ResponseCreatedEvent(ResponseEvent):
    type: Literal["response.created"]
    response: ResponseObject


class ResponseCompletedEvent(ResponseEvent):
    type: Literal["response.completed"]
    response: ResponseObject


class ResponseOutputTextDelta(ResponseEvent):
    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    item_id: str = "item_1234"
    output_index: int = 0
    content_index: int = 0
    delta: str = ""
    logprobs: list = []


class ResponseReasoningSummaryTextDelta(ResponseEvent):
    type: Literal["response.reasoning_summary_text.delta"] = (
        "response.reasoning_summary_text.delta"
    )
    item_id: str = "item_1234"
    output_index: int = 0
    content_index: int = 0
    delta: str = ""


class ResponseReasoningTextDelta(ResponseEvent):
    type: Literal["response.reasoning_text.delta"] = "response.reasoning_text.delta"
    item_id: str = "item_1234"
    output_index: int = 0
    content_index: int = 0
    delta: str = ""


class ResponseReasoningTextDone(ResponseEvent):
    type: Literal["response.reasoning_text.done"] = "response.reasoning_text.done"
    item_id: str = "item_1234"
    output_index: int = 0
    content_index: int = 0
    text: str = ""


class ResponseOutputItemAdded(ResponseEvent):
    type: Literal["response.output_item.added"] = "response.output_item.added"
    output_index: int = 0
    item: Union[
        Item,
        ReasoningItem,
        FunctionCallItem,
        WebSearchCallItem,
        CodeInterpreterCallItem,
    ]


class ResponseOutputItemDone(ResponseEvent):
    type: Literal["response.output_item.done"] = "response.output_item.done"
    output_index: int = 0
    item: Union[
        Item,
        ReasoningItem,
        FunctionCallItem,
        WebSearchCallItem,
        CodeInterpreterCallItem,
    ]


class ResponseInProgressEvent(ResponseEvent):
    type: Literal["response.in_progress"]
    response: ResponseObject


class ResponseContentPartAdded(ResponseEvent):
    type: Literal["response.content_part.added"] = "response.content_part.added"
    item_id: str = "item_1234"
    output_index: int = 0
    content_index: int = 0
    part: Union[TextContentItem, ReasoningTextContentItem]


class ResponseOutputTextDone(ResponseEvent):
    type: Literal["response.output_text.done"] = "response.output_text.done"
    item_id: str = "item_1234"
    output_index: int = 0
    content_index: int = 0
    text: str = ""
    logprobs: list = []


class ResponseContentPartDone(ResponseEvent):
    type: Literal["response.content_part.done"] = "response.content_part.done"
    item_id: str = "item_1234"
    output_index: int = 0
    content_index: int = 0
    part: Union[TextContentItem, ReasoningTextContentItem]


class ResponseOutputTextAnnotationAdded(ResponseEvent):
    type: Literal["response.output_text.annotation.added"] = (
        "response.output_text.annotation.added"
    )
    item_id: str = "item_1234"
    output_index: int = 0
    content_index: int = 0
    annotation_index: int = 0
    annotation: UrlCitation


class ResponseWebSearchCallInProgress(ResponseEvent):
    type: Literal["response.web_search_call.in_progress"] = (
        "response.web_search_call.in_progress"
    )
    output_index: int = 0
    item_id: str = "item_1234"


class ResponseWebSearchCallSearching(ResponseEvent):
    type: Literal["response.web_search_call.searching"] = (
        "response.web_search_call.searching"
    )
    output_index: int = 0
    item_id: str = "item_1234"


class ResponseWebSearchCallCompleted(ResponseEvent):
    type: Literal["response.web_search_call.completed"] = (
        "response.web_search_call.completed"
    )
    output_index: int = 0
    item_id: str = "item_1234"


class ResponseCodeInterpreterCallInProgress(ResponseEvent):
    type: Literal["response.code_interpreter_call.in_progress"] = (
        "response.code_interpreter_call.in_progress"
    )
    output_index: int = 0
    item_id: str = "item_1234"


class ResponseCodeInterpreterCallCompleted(ResponseEvent):
    type: Literal["response.code_interpreter_call.completed"] = (
        "response.code_interpreter_call.completed"
    )
    output_index: int = 0
    item_id: str = "item_1234"
