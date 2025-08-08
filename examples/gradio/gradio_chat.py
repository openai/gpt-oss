import json
import requests
import gradio as gr

DEFAULT_FUNCTION_PROPERTIES = """
{
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
        }
    },
    "required": ["location"]
}
""".strip()

def chat_with_model(message, history, model_choice, instructions, effort, use_functions, 
                   function_name, function_description, function_parameters,
                   use_browser_search, temperature, max_output_tokens, debug_mode):
    
    if not message.strip():
        return history, ""
    
    # Build messages list from history + current message  
    messages = []
    
    # Convert history to messages format
    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append({
                "type": "message",
                "role": "user", 
                "content": [{"type": "input_text", "text": user_msg}]
            })
        if assistant_msg:
            messages.append({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": assistant_msg}]
            })
    
    # Add current message
    messages.append({
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": message}]
    })
    
    # Prepare tools
    tools = []
    if use_functions:
        try:
            tools.append({
                "type": "function",
                "name": function_name,
                "description": function_description,
                "parameters": json.loads(function_parameters),
            })
        except json.JSONDecodeError:
            pass
    
    if use_browser_search:
        tools.append({"type": "browser_search"})
    
    # Get URL based on model (matching streamlit logic)
    url = ("http://localhost:8081/v1/responses" if model_choice == "small" 
           else "http://localhost:8000/v1/responses")
    
    try:
        response = requests.post(
            url,
            json={
                "input": messages,
                "stream": True,
                "instructions": instructions,
                "reasoning": {"effort": effort},
                "metadata": {"__debug": debug_mode},
                "tools": tools,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
            stream=True,
        )
        
        assistant_content = ""
        
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if not data_str:
                continue
            
            try:
                data = json.loads(data_str)
            except Exception:
                continue
            
            event_type = data.get("type", "")
            
            if event_type == "response.reasoning_text.delta":
                assistant_content += data.get("delta", "")
            elif event_type == "response.output_text.delta":
                assistant_content += data.get("delta", "")
            elif event_type == "response.output_item.done":
                item = data.get("item", {})
                if item.get("type") == "function_call":
                    assistant_content += f"\n\n🔨 Called `{item.get('name')}`\n**Arguments**\n```json\n{item.get('arguments', '')}\n```"
                elif item.get("type") == "web_search_call":
                    assistant_content += f"\n\n🌐 **Web Search**\n```json\n{json.dumps(item.get('action', {}), indent=2)}\n```\n✅ Done"
            elif event_type == "response.completed":
                response_data = data.get("response", {})
                if debug_mode:
                    debug_info = response_data.get("metadata", {}).get("__debug", "")
                    if debug_info:
                        assistant_content += f"\n\n**Debug**\n```\n{debug_info}\n```"
                break
        
        new_history = history + [[message, assistant_content]]
        return new_history, ""
        
    except Exception as e:
        error_message = f"❌ Error: {str(e)}"
        new_history = history + [[message, error_message]]
        return new_history, ""


# Create the Gradio interface
with gr.Blocks(title="💬 Chatbot") as demo:
    gr.Markdown("# 💬 Chatbot")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500)
            
            with gr.Row():
                msg = gr.Textbox(placeholder="Type a message...", scale=4, show_label=False)
                send_btn = gr.Button("Send", scale=1)
            
            clear_btn = gr.Button("Clear Chat")
        
        with gr.Column(scale=1):
            model_choice = gr.Radio(["large", "small"], value="small", label="Model")
            
            instructions = gr.Textbox(
                label="Instructions", 
                value="You are a helpful assistant that can answer questions and help with tasks.",
                lines=3
            )
            
            effort = gr.Radio(["low", "medium", "high"], value="medium", label="Reasoning effort")
            
            gr.Markdown("#### Functions")
            use_functions = gr.Checkbox(label="Use functions", value=False)
            
            with gr.Column(visible=False) as function_group:
                function_name = gr.Textbox(label="Function name", value="get_weather")
                function_description = gr.Textbox(
                    label="Function description", 
                    value="Get the weather for a given city"
                )
                function_parameters = gr.Textbox(
                    label="Function parameters", 
                    value=DEFAULT_FUNCTION_PROPERTIES,
                    lines=6
                )
            
            gr.Markdown("#### Built-in Tools") 
            use_browser_search = gr.Checkbox(label="Use browser search", value=False)
            
            temperature = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Temperature")
            max_output_tokens = gr.Slider(1000, 20000, value=1024, step=100, label="Max output tokens")
            
            debug_mode = gr.Checkbox(label="Debug mode", value=False)
    
    # Event handlers
    def toggle_function_group(use_funcs):
        return gr.update(visible=use_funcs)
    
    use_functions.change(toggle_function_group, use_functions, function_group)
    
    # Chat functionality
    inputs = [msg, chatbot, model_choice, instructions, effort, use_functions, 
              function_name, function_description, function_parameters,
              use_browser_search, temperature, max_output_tokens, debug_mode]
    
    msg.submit(chat_with_model, inputs, [chatbot, msg])
    send_btn.click(chat_with_model, inputs, [chatbot, msg])
    clear_btn.click(lambda: [], outputs=chatbot)


if __name__ == "__main__":
    demo.launch()