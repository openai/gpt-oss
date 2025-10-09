from abc import ABC, abstractmethod
from uuid import UUID, uuid4
from typing import AsyncIterator

from openai_harmony import (
    Author,
    Role,
    Message,
    TextContent,
)


def _maybe_update_inplace_and_validate_channel(
    *, input_message: Message, tool_message: Message
) -> None:
    # If the channel of a new message produced by tool is different from the originating message,
    # we auto-set the new message's channel, if unset, or raise an error.
    if tool_message.channel != input_message.channel:
        if tool_message.channel is None:
            tool_message.channel = input_message.channel
        else:
            raise ValueError(
                f"Messages from tool should have the same channel ({tool_message.channel=}) as "
                f"the triggering message ({input_message.channel=})."
            )


class Tool(ABC):
    """
    Something the model can call.

    Tools expose APIs that are shown to the model in a syntax that the model
    understands and knows how to call (from training data). Tools allow the
    model to do things like run code, browse the web, etc.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        An identifier for the tool. The convention is that a message will be routed to the tool
        whose name matches its recipient field.
        """

    @property
    def output_channel_should_match_input_channel(self) -> bool:
        """
        A flag which indicates whether the output channel of the tool should match the input channel.
        """
        return True

    async def process(self, message: Message) -> AsyncIterator[Message]:
        """
        Process the message and return a list of messages to add to the conversation.
        The input message should already be applicable to this tool.
        Don't return the input message, just the new messages.

        If implementing a tool that has to block while calling a function use `call_on_background_thread` to get a coroutine.

        If you just want to test this use `evaluate_generator` to get the results.

        Do not override this method; override `_process` below (to avoid interfering with tracing).
        """
        async for m in self._process(message):
            if self.output_channel_should_match_input_channel:
                _maybe_update_inplace_and_validate_channel(input_message=message, tool_message=m)
            yield m

    @abstractmethod
    async def _process(self, message: Message) -> AsyncIterator[Message]:
        """Override this method to provide the implementation of the tool."""
        if False:  # This is to convince the type checker that this is an async generator.
            yield  # type: ignore[unreachable]
        _ = message  # Stifle "unused argument" warning.
        raise NotImplementedError

    @abstractmethod
    def instruction(self) -> str:
        """
        Describe the tool's functionality. For example, if it accepts python-formatted code,
        provide documentation on the functions available.
        """
        raise NotImplementedError

    def instruction_dict(self) -> dict[str, str]:
        return {self.name: self.instruction()}

    def error_message(
        self, error_message: str, id: UUID | None = None, channel: str | None = None
    ) -> Message:
        """
        Return an error message that's from this tool.
        """
        return Message(
            id=id if id else uuid4(),
            author=Author(role=Role.TOOL, name=self.name),
            content=TextContent(text=error_message), # TODO: Use SystemError instead
            channel=channel,
        ).with_recipient("assistant")

