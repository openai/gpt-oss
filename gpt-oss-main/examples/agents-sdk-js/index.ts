import { OpenAI } from "openai";
import {
  Agent,
  run,
  setDefaultOpenAIClient,
  setOpenAIAPI,
  setTracingDisabled,
  tool,
  MCPServerStdio,
} from "@openai/agents";
import { z } from "zod";
import path from "node:path";
import process from "node:process";
import { styleText } from "node:util";
import { createInterface } from "node:readline/promises";

async function prompt(question: string) {
  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  const answer = await rl.question(question);
  rl.close();
  return answer;
}

const openai = new OpenAI({
  apiKey: "local",
  baseURL: "http://localhost:11434/v1",
});

const samplesDir = path.join(process.cwd());

const mcpServer = new MCPServerStdio({
  name: "Filesystem MCP Server, via npx",
  fullCommand: `npx -y @modelcontextprotocol/server-filesystem ${samplesDir}`,
});

await mcpServer.connect();

setTracingDisabled(true);
setDefaultOpenAIClient(openai);
setOpenAIAPI("chat_completions");

const searchTool = tool({
  name: "get_current_weather",
  description: "Get the current weather in a given location",
  parameters: z.object({
    location: z.string(),
  }),
  execute: async ({ location }) => {
    return `The weather in ${location} is sunny.`;
  },
});

const agent = new Agent({
  name: "My Agent",
  instructions: "You are a helpful assistant.",
  tools: [searchTool],
  model: "gpt-oss:20b-test",
  mcpServers: [mcpServer],
});

const input = await prompt("> ");

const result = await run(agent, input, {
  stream: true,
});

for await (const event of result) {
  if (event.type === "raw_model_stream_event" && event.data.type === "model") {
    if (event.data.event.choices[0].delta.content) {
      process.stdout.write(event.data.event.choices[0].delta.content);
    } else if (event.data.event.choices[0].delta.reasoning) {
      process.stdout.write(event.data.event.choices[0].delta.reasoning);
    }
  } else if (
    event.type === "run_item_stream_event" &&
    event.item.type === "tool_call_item" &&
    event.item.rawItem.type == "function_call"
  ) {
    console.log(
      `\nCalling ${event.item.rawItem.name} with: ${event.item.rawItem.arguments}`
    );
  }
}

console.log("\n");
await result.completed;
await mcpServer.close();
