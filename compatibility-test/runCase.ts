import {
  Agent,
  Runner,
  OpenAIResponsesModel,
  OpenAIChatCompletionsModel,
  RunResult,
  StreamedRunResult,
  FunctionTool,
  setTracingDisabled,
} from "@openai/agents";
import { Ajv } from "ajv";
import { OpenAI } from "openai";
import { PROVIDERS } from "./providers";
import { TOOLS_MAP } from "./tools";

setTracingDisabled(true);

const ajv = new Ajv();

export type Case = {
  tool_name: string;
  input: string;
  expected_arguments: string;
  instructions?: string;
};

// Summary shape for each apiType
export type RunCaseSummary = {
  apiType: string;
  success: boolean;
  validResponse: boolean;
  validEvents?: boolean;
  details: Record<string, any>;
  history: any[];
  successToolCall: boolean;
  toolCallingDetails: Record<string, any>;
};

export async function runCase(
  provider: string,
  caseData: Case,
  {
    maxTurns,
    streaming,
    strict,
  }: { maxTurns: number; streaming: boolean; strict: boolean }
): Promise<RunCaseSummary[]> {
  const config = PROVIDERS[provider];
  if (!config) {
    throw new Error(
      `Provider ${provider} not found. Valid providers are: ${Object.keys(
        PROVIDERS
      ).join(", ")}`
    );
  }

  const agent = new Agent({
    name: caseData.tool_name,
    instructions: caseData.instructions,
    tools: [TOOLS_MAP[caseData.tool_name]],
  });

  const client = new OpenAI({
    apiKey: config.apiKey,
    baseURL: config.apiBaseUrl,
  });

  const summaries: RunCaseSummary[] = [];

  for (const apiType of config.apiType) {
    const runner = new Runner({
      model:
        apiType === "responses"
          ? new OpenAIResponsesModel(client, config.modelName)
          : new OpenAIChatCompletionsModel(client, config.modelName),
      modelSettings: {
        providerData: config.providerDetails ?? {},
      },
    });

    let result: RunResult<any, any> | StreamedRunResult<any, any>;
    let streamedEvents: any[] | undefined = undefined;
    if (streaming) {
      result = await runner.run(agent, caseData.input, {
        stream: streaming,
        maxTurns: maxTurns,
      });
      if (result instanceof StreamedRunResult) {
        // Collect streaming events if applicable
        streamedEvents = [];
        for await (const event of result) {
          if (event.type === "raw_model_stream_event") {
            if (event.data.type === "model") {
              streamedEvents.push(event.data.event);
            }
          }
        }
        await result.completed;
      }
    } else {
      result = await runner.run(agent, caseData.input, {
        maxTurns: maxTurns,
      });
    }

    const { success: successToolCall, details: toolCallingDetails } =
      testToolCall(apiType, caseData, result, strict);

    const { validResponse, details } = testOutputData(
      apiType,
      result.rawResponses,
      streaming
    );

    const { validEvents, details: eventsDetails } = streaming
      ? testEvents(apiType, streamedEvents)
      : { validEvents: true, details: {} };

    let success = successToolCall && validResponse;
    if (streaming) {
      success = success && validEvents;
    }
    const summary: RunCaseSummary = {
      apiType,
      success,
      validResponse,
      validEvents,
      details: {
        ...details,
        ...eventsDetails,
      },
      history: result?.rawResponses.map((entry) => entry.providerData) ?? [],
      successToolCall,
      toolCallingDetails,
    };

    summaries.push(summary);
  }

  return summaries;
}

function testToolCall(apiType, caseData, result, strict) {
  let details: Record<string, boolean | string> = {};
  result.newItems.forEach((item) => {
    // for this test for now we only care if the tool is called at least once
    if (details.calledToolAtLeastOnce) {
      return;
    }

    const isToolCall = item.type === "tool_call_item";
    if (isToolCall) {
      if (item.rawItem.type === "function_call") {
        if (item.rawItem.name === caseData.tool_name) {
          const validate = ajv.compile(
            (TOOLS_MAP[caseData.tool_name] as FunctionTool).parameters
          );
          const valid = validate(JSON.parse(item.rawItem.arguments));
          details.calledToolWithRightSchema = valid;
          details.calledToolAtLeastOnce = true;

          if (details.calledToolWithRightSchema) {
            const parsedArguments = JSON.parse(item.rawItem.arguments);
            const expectedArguments = JSON.parse(caseData.expected_arguments);
            details.calledToolWithRightArguments = deepEqual(
              parsedArguments,
              expectedArguments
            );
            if (!details.calledToolWithRightArguments) {
              if (details.calledToolWithRightSchema) {
                details.warning = `Tool call with wrong arguments but correct schema. Check logs for full details. Not failing this test. Parsed: ${JSON.stringify(
                  parsedArguments
                )} Expected: ${JSON.stringify(expectedArguments)}`;
              }
              details.actualArguments = parsedArguments;
              details.expectedArguments = expectedArguments;
            }
          }
        }
      }
    }
  });

  return {
    success:
      !!details.calledToolAtLeastOnce &&
      !!details.calledToolWithRightSchema &&
      (!strict || !!details.calledToolWithRightArguments),
    details,
  };
}

function testEvents(apiType, events) {
  // In an ideal world we would check all the events to follow and reconstruct the final response
  // and then compare it against the final response in the response.completed event
  // for now we just check that certain events are present

  let details: Record<string, boolean> = {};
  let validEvents: boolean = false;

  if (apiType === "chat") {
    let hasReasoningDeltas = false;
    for (const event of events) {
      hasReasoningDeltas =
        hasReasoningDeltas ||
        (typeof event.choices[0].delta.reasoning === "string" &&
          event.choices[0].delta.reasoning.length > 0);
    }
    details.hasReasoningDeltas = hasReasoningDeltas;
    validEvents = hasReasoningDeltas;
  }

  if (apiType === "responses") {
    let hasReasoningDeltaEvents = false;
    let hasReasoningDoneEvents = false;
    for (const event of events) {
      if (event.type === "raw_model_stream_event") {
        if (event.data.type === "model") {
          if (event.data.event.type === "response.reasoning_text.delta") {
            hasReasoningDeltaEvents = true;
          }
          if (event.data.event.type === "response.reasoning_text.done") {
            hasReasoningDoneEvents = true;
          }
        }
      }
    }

    details.hasReasoningDeltaEvents = hasReasoningDeltaEvents;
    details.hasReasoningDoneEvents = hasReasoningDoneEvents;
    validEvents =
      details.hasReasoningDeltaEvents && details.hasReasoningDoneEvents;
  }

  return {
    validEvents,
    details,
  };
}

function testOutputData(apiType, rawResponses, streaming) {
  let details: Record<string, boolean> = {};
  let validResponse: boolean = false;

  if (apiType === "chat") {
    for (const response of rawResponses) {
      if (streaming && !response.providerData) {
        // with Chat Completions we don't have a final response object that's native so we skip this test
        return {
          validResponse: true,
          details: {
            skippedBecauseStreaming: true,
          },
        };
      }

      // this is the actual HTTP response from the provider
      // Since it's not guaranteed that every response has a reasoning field, we check if it's present
      // at least once across all responses
      const data = response.providerData;
      const message = data.choices[0].message;
      if (message.role === "assistant" && !message.refusal) {
        details.hasReasoningField =
          details.hasReasoningField ||
          ("reasoning" in message && typeof message.reasoning === "string");
        details.hasReasoningContentField =
          details.hasReasoningContentField ||
          ("reasoning_content" in message &&
            typeof message.reasoning_content === "string");

        validResponse =
          validResponse ||
          (details.hasReasoningField && message.reasoning.length > 0);
      }
    }
  } else if (apiType === "responses") {
    // this is the actual HTTP response from the provider
    const data = rawResponses[0].providerData;
    for (const item of data.output) {
      // Since it's not guaranteed that every response has a reasoning field, we check if it's present
      // at least once across all responses

      if (item.type === "reasoning") {
        details.hasReasoningContentArray = Array.isArray(item.content);
        details.hasReasoningContentArrayLength = item.content.length > 0;
        details.hasReasoningContentArrayItemType = item.content.every(
          (item) => item.type === "reasoning_text"
        );
        details.hasReasoningContentArrayItemText = item.content.every(
          (item) => item.text.length > 0
        );

        validResponse =
          details.hasReasoningContentArray &&
          details.hasReasoningContentArrayLength &&
          details.hasReasoningContentArrayItemType &&
          details.hasReasoningContentArrayItemText;
      }
    }
  }

  return {
    validResponse,
    details,
  };
}

function deepEqual(a: any, b: any): boolean {
  if (a === b) return true;
  if (typeof a !== typeof b) return false;
  if (a && b && typeof a === "object") {
    if (Array.isArray(a) !== Array.isArray(b)) return false;
    if (Array.isArray(a)) {
      if (a.length !== b.length) return false;
      for (let i = 0; i < a.length; i++) {
        if (!deepEqual(a[i], b[i])) return false;
      }
      return true;
    } else {
      const aKeys = Object.keys(a);
      const bKeys = Object.keys(b);
      if (aKeys.length !== bKeys.length) return false;
      for (const key of aKeys) {
        if (!b.hasOwnProperty(key)) return false;
        if (!deepEqual(a[key], b[key])) return false;
      }
      return true;
    }
  }
  return false;
}
