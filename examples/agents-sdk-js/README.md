# JavaScript Agents SDK Example

This example demonstrates how to use GPT-OSS with OpenAI's Agents SDK in TypeScript/JavaScript.

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- GPT-OSS model running locally (Ollama, vLLM, etc.)
- npm or yarn package manager

### Installation

1. **Install dependencies:**
```bash
npm install
```

2. **Install global dependencies (for MCP server):**
```bash
npm install -g npx
```

### Running the Example

1. **Start your GPT-OSS model:**
```bash
# Using Ollama
ollama run gpt-oss:20b

# Using vLLM
vllm serve openai/gpt-oss-20b --port 11434
```

2. **Run the example:**
```bash
npm start
```

## 🔧 Configuration

### Environment Setup
The example is configured to use a local model server:

```typescript
const openai = new OpenAI({
  apiKey: "local",
  baseURL: "http://localhost:11434/v1",
});
```

### Model Configuration
```typescript
const agent = new Agent({
  name: "My Agent",
  instructions: "You are a helpful assistant.",
  tools: [searchTool],
  model: "gpt-oss:20b-test",  // Model name for local server
  mcpServers: [mcpServer],
});
```

## 🛠️ Features Demonstrated

### Function Calling
The example includes a weather tool:
```typescript
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
```

### MCP (Model Context Protocol) Integration
Filesystem access via MCP server:
```typescript
const mcpServer = new MCPServerStdio({
  name: "Filesystem MCP Server, via npx",
  fullCommand: `npx -y @modelcontextprotocol/server-filesystem ${samplesDir}`,
});
```

### Streaming Responses
Real-time response streaming:
```typescript
const result = await run(agent, input, {
  stream: true,
});

for await (const event of result) {
  // Process streaming events
}
```

## 📝 Code Structure

### Main Components
1. **Client Setup**: OpenAI client configuration for local model
2. **MCP Server**: Filesystem access server
3. **Tool Definition**: Custom function calling tool with Zod validation
4. **Agent Creation**: GPT-OSS agent with tools and MCP
5. **Streaming Execution**: Real-time response processing

### Event Types
- `raw_model_stream_event`: Raw model responses and reasoning
- `run_item_stream_event`: Tool calls and function executions

### TypeScript Features
- **Zod Validation**: Type-safe parameter validation
- **Async/Await**: Modern JavaScript async patterns
- **Type Safety**: Full TypeScript support

## 🐛 Troubleshooting

### Common Issues

1. **"npx is not installed"**
   ```bash
   npm install -g npx
   ```

2. **Connection refused to localhost:11434**
   - Ensure your model server is running
   - Check the port number matches your setup

3. **TypeScript compilation errors**
   ```bash
   # Check TypeScript version
   npx tsc --version
   
   # Install missing types
   npm install @types/node
   ```

4. **Module resolution errors**
   ```bash
   # Clear npm cache
   npm cache clean --force
   
   # Reinstall dependencies
   rm -rf node_modules package-lock.json
   npm install
   ```

### Debug Mode
Enable verbose logging:
```typescript
// Add to your code
console.log('Event:', event);
```

## 📦 Package Scripts

- `npm start`: Run the example with tsx
- `npm test`: Run tests (placeholder)
- `npx tsc`: Compile TypeScript
- `npx tsx index.ts`: Run directly with tsx

## 🔗 Related Documentation

- [OpenAI Agents SDK](https://github.com/openai/agents) - Official SDK documentation
- [Model Context Protocol](https://modelcontextprotocol.io/) - MCP specification
- [Zod](https://zod.dev/) - TypeScript-first schema validation
- [tsx](https://github.com/esbuild-kit/tsx) - TypeScript execution engine
- [Main Examples README](../README.md) - Overview of all examples

## 🤝 Contributing

Improvements welcome! Please:
- Add more tool examples
- Enhance error handling
- Add configuration options
- Improve TypeScript types
- Add unit tests
