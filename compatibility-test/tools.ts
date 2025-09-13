import { Tool, tool } from "@openai/agents";

function convertToTool(toolData: any) {
  return tool({
    name: toolData.name,
    description: toolData.description,
    parameters: toolData.parameters,
    execute: async (parameters) => {
      return toolData.output;
    },
    strict: false,
  });
}

export const TOOLS = [
  {
    type: "function",
    name: "get_weather",
    description: "Get the weather for a given location",
    parameters: {
      type: "object",
      properties: {
        location: {
          type: "string",
          description: "The location to get the weather for",
        },
      },
      required: ["location"],
      additionalProperties: false,
    },
    output: '{"weather":"sunny"}',
  },
  {
    type: "function",
    name: "get_system_health",
    description:
      "Returns the current health status of the LLM runtime—use before critical operations to verify the service is live.",
    parameters: { type: "object", properties: {} },
    output: '{"status":"ok","uptime_seconds":372045}',
  },
  {
    type: "function",
    name: "markdown_to_html",
    description:
      "Converts a Markdown string to sanitized HTML—use when you need browser-renderable output.",
    parameters: {
      type: "object",
      properties: {
        markdown: { type: "string", description: "Raw Markdown content" },
      },
      required: ["markdown"],
      additionalProperties: false,
    },
    output: '{"html":"<h1>Hello World</h1><p>This is <em>great</em>.</p>"}',
  },
  {
    type: "function",
    name: "detect_language",
    description:
      "Identifies the ISO language code of the supplied text—use for routing text to language-specific models.",
    parameters: {
      type: "object",
      properties: {
        text: {
          type: "string",
          description: "Text whose language should be detected",
        },
      },
      required: ["text"],
      additionalProperties: false,
    },
    output: '{"language":"de","confidence":0.98}',
  },
  {
    type: "function",
    name: "generate_chart",
    description:
      "Creates a base64-encoded PNG chart from tabular data—use for quick visualizations inside chat.",
    parameters: {
      type: "object",
      properties: {
        data: {
          type: "array",
          items: { type: "array", items: { type: "number" } },
          description: "2-D numeric data matrix",
        },
        chart_type: {
          type: "string",
          enum: ["line", "bar", "scatter"],
          description: "Type of chart to generate",
        },
        title: {
          type: "string",
          description: "Chart title",
          default: "",
        },
        x_label: {
          type: "string",
          description: "Label for the x-axis",
          default: "",
        },
        y_label: {
          type: "string",
          description: "Label for the y-axis",
          default: "",
        },
      },
      required: ["data", "chart_type"],
      additionalProperties: false,
    },
    output: '{"image_png_base64":"iVBORw0KGgoAAAANSUhEUgAA..."}',
  },
  {
    type: "function",
    name: "query_database",
    description:
      "Runs a parameterized SQL SELECT on the internal analytics DB—use for lightweight data look-ups.",
    parameters: {
      type: "object",
      properties: {
        table: { type: "string", description: "Table name to query" },
        columns: {
          type: "array",
          items: { type: "string" },
          description: "Columns to return",
        },
        filters: {
          type: "string",
          description: "SQL WHERE clause without the word WHERE",
          default: "",
        },
        limit: {
          type: "integer",
          minimum: 1,
          maximum: 10000,
          description: "Max rows to return",
          default: 100,
        },
        order_by: {
          type: "string",
          description: "Column to order by (optional)",
          default: "",
        },
      },
      required: ["table", "columns"],
      additionalProperties: false,
    },
    output:
      '{"rows":[{"id":1,"email":"user@example.com"},{"id":2,"email":"foo@bar.com"}],"row_count":2}',
  },
];

export const TOOLS_MAP = TOOLS.reduce((acc, tool) => {
  acc[tool.name] = convertToTool(tool);
  return acc;
}, {} as Record<string, Tool>);
