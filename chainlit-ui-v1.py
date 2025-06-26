import json
import os
from typing import Any, List, Dict, Union, Optional
import base64

import asyncio
import chainlit as cl
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from chainlit.element import Element
from datetime import datetime
from models.openai_model import OpenAIModel
from models.ollama_model import OllamaModel
from MCPClient import MCPClient
from MCPServerManager import MCPServerManager
model=None

# System prompt that guides the LLM's behavior and capabilities
SYSTEM_PROMPT = """You are a helpful assistant capable of accessing external functions from multiple servers and engaging in casual chat. Use the responses from these function calls to provide accurate and informative answers. Be natural and hide the fact that you are using tools to access real-time information. Guide the user about available tools and their capabilities. Always utilize tools to access real-time information when required.

# Tools

{tools}

# Document Management
You can help users manage documents across multiple repositories:
- Upload documents via the file upload feature
- View document catalog from all connected servers
- Retrieve document details from any server
- Delete documents when no longer needed

# Notes

- Ensure responses are based on the latest information available from function calls.
- Maintain an engaging, supportive, and friendly tone throughout the dialogue.
- Always highlight the potential of available tools to assist users comprehensively."""

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create the server manager globally
server_manager = MCPServerManager()
conversation_messages = None


@cl.action_callback("show_conversion_tab")
async def show_conversion_tab(action):
    """Show the document conversion interface using Pandoc"""
    # Create a message with the conversion interface
    msg = cl.Message(content="# Document Format Conversion\n\nConvert your documents between various formats using Pandoc.")
    await msg.send()

    # Step 1: Check if Pandoc server is available
    pandoc_server = server_manager.servers.get("Pandoc Server")
    if not pandoc_server:
        await cl.Message(content="⚠️ Pandoc Server is not available. Please restart the chat and try again.").send()
        return

    # Get conversion tool
    if "convert_document" not in pandoc_server.tools:
        await cl.Message(content="⚠️ The Pandoc Server doesn't have a convert_document tool.").send()
        return

    # Step 2: Get list of all documents from all servers
    document_list = []
    for server_name, client in server_manager.servers.items():
        if "list_documents" in client.tools:
            try:
                list_tool = client.tools["list_documents"]["callable"]
                documents = await list_tool(limit=50)  # Get up to 50 documents
                print("all documents:",documents)
                # Parse JSON response
                try:
                    docs = json.loads(documents)
                    for doc in docs:
                        # Add server name to each document
                        doc["server_name"] = server_name
                        document_list.append(doc)
                        print("Document list:",document_list)
                except:
                    continue
            except Exception as e:
                print(f"Error listing documents from {server_name}: {e}")

    if not document_list:
        await cl.Message(content="No documents found. Please upload some documents first.").send()
        return

    # Step 3: Create document selection interface
    doc_actions = []
    for i, doc in enumerate(document_list[:10]):  # Limit to first 10 docs
        doc_id = doc.get("id", "unknown")
        doc_name = doc.get("filename", f"Document {doc_id[:8]}")
        server = doc.get("server_name", "unknown")
        
        # Create action for this document
        doc_action = cl.Action(
            name=f"convert_doc_{i}_{doc_id[:8]}",
            label=f"Convert: {doc_name}",
            description=f"From {server}",
            type="action",
            payload={"doc_id": doc_id, "server_name": server, "doc_name": doc_name}
        )
        doc_actions.append(doc_action)

    # Show document selection actions in groups of 3
    for i in range(0, len(doc_actions), 3):
        chunk = doc_actions[i:i+3]
        await cl.Message(content="", actions=chunk).send()

    # Add a "more documents" action if needed
    if len(document_list) > 10:
        more_action = cl.Action(
            name="more_documents",
            label="Show More Documents",
            description="Display additional documents",
            type="action",
            payload={}
        )
        await cl.Message(content="", actions=[more_action]).send()

    # Add an action to go back to the main interface
    back_action = cl.Action(
        name="view_activity",  # Reuse existing action
        label="⬅️ Back to Activity",
        description="Return to activity view",
        type="action",
        payload={}
    )
    await cl.Message(content="", actions=[back_action]).send()

    # Register callbacks for document selection
    for i, doc in enumerate(document_list[:10]):
        doc_id = doc.get("id", "unknown")
        
        @cl.action_callback(f"convert_doc_{i}_{doc_id[:8]}")
        async def on_doc_select(action, doc_id=doc_id):
            await show_conversion_options(action.payload)

@cl.action_callback("more_documents")
async def show_more_documents(action):
    # Implement pagination for documents
    # This is a simplified version - would need to keep track of page number
    await cl.Message(content="This feature is not yet implemented. Please use one of the documents shown.").send()

async def show_conversion_options(doc_payload):
    """Show format conversion options for the selected document"""
    doc_id = doc_payload.get("doc_id")
    server_name = doc_payload.get("server_name")
    doc_name = doc_payload.get("doc_name", f"Document {doc_id[:8]}")
    
    # Show a message about the selected document
    msg = cl.Message(content=f"Selected document: **{doc_name}**\n\nChoose target format:")
    await msg.send()
    
    # Define supported format conversions
    formats = [
        {"name": "PDF", "value": "pdf"},
        {"name": "Word (DOCX)", "value": "docx"},
        {"name": "Markdown", "value": "markdown"},
        {"name": "HTML", "value": "html"},
        {"name": "Plain Text", "value": "txt"},
        {"name": "Rich Text Format", "value": "rtf"},
        {"name": "OpenDocument Text", "value": "odt"},
        {"name": "LaTeX", "value": "latex"},
        {"name": "EPUB", "value": "epub"}
    ]
    
    # Create actions for each format
    format_actions = []
    for fmt in formats:
        format_action = cl.Action(
            name=f"convert_to_{fmt['value']}",
            label=f"Convert to {fmt['name']}",
            description=f"Convert document to {fmt['name']} format",
            type="action",
            payload={
                "doc_id": doc_id,
                "server_name": server_name,
                "target_format": fmt["value"],
                "format_name": fmt["name"],
                "doc_name": doc_name
            }
        )
        format_actions.append(format_action)
    
    # Show format selection actions in groups of 3
    for i in range(0, len(format_actions), 3):
        chunk = format_actions[i:i+3]
        await cl.Message(content="", actions=chunk).send()
    
    # Add a cancel action
    cancel_action = cl.Action(
        name="cancel_conversion",
        label="Cancel",
        description="Cancel the conversion",
        type="action",
        payload={}
    )
    await cl.Message(content="", actions=[cancel_action]).send()
    
    # Register callbacks for format selection
    for fmt in formats:
        @cl.action_callback(f"convert_to_{fmt['value']}")
        async def on_format_select(action):
            await perform_conversion(action.payload)

@cl.action_callback("cancel_conversion")
async def cancel_conversion(action):
    """Cancel the conversion process and return to the main interface"""
    await show_conversion_tab(action)

async def perform_conversion(conversion_payload):
    """Perform the actual document conversion using Pandoc"""
    doc_id = conversion_payload.get("doc_id")
    server_name = conversion_payload.get("server_name")
    target_format = conversion_payload.get("target_format")
    format_name = conversion_payload.get("format_name")
    doc_name = conversion_payload.get("doc_name")
    
    # Show a loading message
    msg = cl.Message(content=f"Converting **{doc_name}** to {format_name}...")
    await msg.send()
    
    try:
        # Step 1: Get the document content from its server
        source_client = server_manager.servers.get(server_name)
        if not source_client or "display_document" not in source_client.tools:
            msg.content = f"⚠️ Error: Could not access document on {server_name} server"
            await msg.update()
            return
            
        display_tool = source_client.tools["display_document"]["callable"]
        document_data = await display_tool(doc_id=doc_id, include_original=True)
        
        # Parse the returned document data
        try:
            doc_data = json.loads(document_data)
            # Prefer original content if available, otherwise use markdown
            content = doc_data.get("original_content", doc_data.get("markdown_content", ""))
            metadata = doc_data.get("metadata", {})
            source_format = metadata.get("original_format", "md").lower()
        except (json.JSONDecodeError, AttributeError):
            # If not JSON, assume it's just the content
            content = document_data
            source_format = "md"  # Default to markdown
        
        # Step 2: Get Pandoc server and conversion tool
        pandoc_server = server_manager.servers.get("Pandoc Server")
        if not pandoc_server or "convert_document" not in pandoc_server.tools:
            msg.content = "⚠️ Pandoc Server is not available for conversion"
            await msg.update()
            return
            
        convert_tool = pandoc_server.tools["convert_document"]["callable"]
        
        # Create a step for the conversion process
        conversion_step = cl.Step(
            name=f"Converting {doc_name} to {format_name}",
            type="tool",
            parent_id=msg.id
        )
        await conversion_step.send()
        
        # Step 3: Perform the conversion
        converted_content = await convert_tool(
            content=content,
            from_format=source_format,
            to_format=target_format,
            options=json.dumps({
                "standalone": True,
                "preserve_tabs": True,
                "wrap": "auto"
            })
        )
        
        conversion_step.content = f"Successfully converted document from {source_format} to {target_format}"
        await conversion_step.update()
        
        # Step 4: Store the converted document
        # Find a document server to store the result
        doc_server = server_manager.servers.get("Document Server")
        if not doc_server or "ingest_document" not in doc_server.tools:
            msg.content = "⚠️ Document Server is not available to store the converted document"
            await msg.update()
            return
            
        ingest_tool = doc_server.tools["ingest_document"]["callable"]
        
        # Create a base name for the converted document
        base_name = os.path.splitext(doc_name)[0]
        new_filename = f"{base_name}.{target_format}"
        
        # Prepare metadata
        new_metadata = {
            "filename": new_filename,
            "original_size": len(content),
            "converted_size": len(converted_content) if isinstance(converted_content, str) else 0,
            "original_format": source_format,
            "target_format": target_format,
            "converted_at": datetime.now().isoformat(),
            "original_doc_id": doc_id,
            "original_server": server_name,
            "conversion_type": "pandoc"
        }
        
        # Generate summary
        summary = f"Converted from {source_format} to {target_format} - Original: {doc_name}"
        
        # Store the converted document
        storage_step = cl.Step(
            name=f"Storing converted document: {new_filename}",
            type="tool",
            parent_id=msg.id
        )
        await storage_step.send()
        
        new_doc_id = await ingest_tool(
            content=converted_content,
            doc_type=target_format,
            entities=json.dumps(new_metadata),
            summary=summary
        )
        
        storage_step.content = f"Stored converted document with ID: {new_doc_id}"
        await storage_step.update()
        
        # Update the message with success
        msg.content = (
            f"✅ Successfully converted **{doc_name}** to {format_name}\n\n"
            f"📝 **Details:**\n"
            f"- Original format: {source_format.upper()}\n"
            f"- New format: {format_name}\n"
            f"- New document ID: {new_doc_id}\n"
            f"- New filename: {new_filename}\n\n"
            f"You can view the converted document using:\n"
            f"`/display {new_doc_id} on Document Server`"
        )
        await msg.update()
        
        # Add view document action
        view_action = cl.Action(
            name=f"view_converted_doc_{new_doc_id[:8]}",
            label="📄 View Converted Document",
            description=f"View document with ID: {new_doc_id}",
            type="action",
            payload={"doc_id": new_doc_id, "server_name": "Document Server"}
        )
        
        # Add convert another action
        convert_another_action = cl.Action(
            name="show_conversion_tab",
            label="🔄 Convert Another Document",
            description="Convert another document",
            type="action",
            payload={}
        )
        
        await cl.Message(content="", actions=[view_action, convert_another_action]).send()
        
        # Register callback for viewing the converted document
        @cl.action_callback(f"view_converted_doc_{new_doc_id[:8]}")
        async def view_converted_doc(action):
            await display_document(action.payload["doc_id"], action.payload["server_name"])
            
    except Exception as e:
        error_message = f"Error during conversion: {str(e)}"
        print(f"Conversion error details: {type(e).__name__}: {str(e)}")
        
        if 'conversion_step' in locals() and not conversion_step.status == "error":
            conversion_step.content = error_message
            conversion_step.status = "error"
            await conversion_step.update()
        
        if 'storage_step' in locals() and not storage_step.status == "error":
            storage_step.content = error_message
            storage_step.status = "error"
            await storage_step.update()
        
        msg.content = f"❌ Conversion failed: {error_message}"
        await msg.update()
        
        # Add try again action
        retry_action = cl.Action(
            name="show_conversion_tab",
            label="🔄 Try Again",
            description="Return to conversion interface",
            type="action",
            payload={}
        )
        await cl.Message(content="", actions=[retry_action]).send()

# Add this to the on_chat_start function to create the conversion button
def add_conversion_button():
    conversion_action = cl.Action(
        name="show_conversion_tab",
        label="🔄 Convert Documents",
        description="Convert documents between formats",
        type="action",
        payload={}
    )
    return conversion_action


@cl.on_chat_start
async def on_chat_start():
    global server_manager, conversation_messages, model
    
     # Initialize the model
    try:
        model = OpenAIModel(model_id="gpt-4")  # or OllamaModel(model_name="llama2")
        await model.initialize()
    except Exception as e:
        error_msg = f"Error initializing model: {str(e)}"
        await cl.Message(content=error_msg).send()
        return

    # Initialize conversation history
    conversation_messages = None
    
    # Show a welcome message
    welcome_msg = cl.Message(content="Setting up multiple MCP servers. Please wait...")
    await welcome_msg.send()
    
    # Configure MCP servers
    server_configs = [
        {
            "name": "Document Server",
            "command": "docker",
            "args": [
                "run",
                "--rm",
                "-i",
                "-v",
                "mcp-documents:/app/documents",
                "mcp/document-ingestion",
                "--storage-path",
                "/app/documents",
                "--mongo-uri",
                "mongodb://host.docker.internal:27017",
                "--db-name",
                "mcp_documents",
            ],
            "env": None,
        },
        {
            "name": "SQLite Server",
            "command": "docker",
            "args": [
                "run",
                "--rm",  # Remove container after exit
                "-i",  # Interactive mode
                "-v",  # Mount volume
                "mcp-test:/mcp",  # Map local volume to container path
                "mcp/sqlite",  # Use SQLite MCP image
                "--db-path",
                "/mcp/test.db",  # Database file path inside container
            ],
            "env": None,
        }, {
        "name": "Markitdown Server",
        "command": "docker",
        "args": [
            "run",
            "--rm",
            "-i",
            "-v",  # Mount volume
            "mcp-test:/mcp",
            "markitdown-mcp-server",
            
        ],
        "env": None,
    }
    ]
    
    # Start MCP clients for each server
    connected_servers = []
    failed_servers = []
    
    for server_config in server_configs:
        welcome_msg.content = f"Connecting to {server_config['name']}..."
        await welcome_msg.update()
        
        success = await server_manager.add_server(server_config)
        if success:
            connected_servers.append(server_config["name"])
        else:
            failed_servers.append(server_config["name"])
    
    # Prepare message about connected servers
    if connected_servers:
        server_list = ", ".join(connected_servers)
        tools_count = len(server_manager.all_tools)
        
        # Get resources from all servers
        all_resources = await server_manager.get_all_resources()
        resource_names = [f"{r.name} ({r.server_name})" for r in all_resources]
        
       
        
        success_message = f"Connected to {len(connected_servers)} MCP servers: {server_list}\n\nFound {tools_count} tools across all servers\n\nYou can upload documents using the file upload button."
    else:
        success_message = "Failed to connect to any MCP servers. Please check configurations and try again."
    
    # Report on failed servers
    if failed_servers:
        failed_list = ", ".join(failed_servers)
        success_message += f"\n\nFailed to connect to: {failed_list}"
    
    # Update the welcome message with results
    welcome_msg.content = success_message
    await welcome_msg.update()

    # Create action buttons
    if connected_servers:
        activity_action = cl.Action(
            name="view_activity",
            label="📊 View Activity",
            description="Shows recent document activity",
            type="action",
            payload={}
        )
              
        # Import the function from our added code
        
        conversion_action = add_conversion_button()
        
        # Add both buttons to the interface
        await cl.Message(content="", actions=[activity_action, conversion_action]).send()



async def agent_loop(query: str, tools: Dict, messages: List[Dict] = None, msg=None):
    """
    Main interaction loop that processes user queries using the LLM and available tools.
    """
    if messages is None:
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    tools="\n- ".join(
                        [
                            f"{t['name']} ({t['server_name']}): {t['schema']['function']['description']}"
                            for t in tools.values()
                        ]
                    )
                ),
            },
        ]

    # Add user query to the messages list
    messages.append({"role": "user", "content": query})
    
    # Show thinking indicator
    if msg:
        # Update message content in Chainlit-compatible way
        msg.content = "Thinking..."
        await msg.update()

    # Query LLM with the system prompt, user query, and available tools
    try:
        first_response = await model.generate_response(
            messages=messages,
            tools=[t["schema"] for t in tools.values()] if tools else None
        )
    except Exception as e:
        error_msg = f"Error during OpenAI API call: {e}"
        if msg:
            msg.content = error_msg
            await msg.update()
        return error_msg, messages

    if hasattr(first_response, 'choices'):
        # OpenAI-style response
        response_message = first_response.choices[0].message
        message_dict = {
            "role": "assistant",
            "content": response_message.content or ""
        }
        if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
            message_dict["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                } for tool_call in response_message.tool_calls
            ]
    else:
        # Handle dictionary response format
        message_dict = {
            "role": "assistant", 
            "content": first_response.get("content", "")
        }
        # Add tool_calls if present
        if first_response.get("tool_calls"):
            message_dict["tool_calls"] = first_response["tool_calls"]
    messages.append(message_dict)


    if message_dict.get("tool_calls"):
        if msg:
            msg.content = "Calling tools across servers..."
            await msg.update()
        
        tool_results = []
        
        # Process tool calls
        for tool_call in message_dict["tool_calls"]:
            function_name = tool_call["function"]["name"]
            
            # Find the matching tool in our tools dictionary
            matching_tool = None
            for tool_key, tool_info in tools.items():
                if tool_info["schema"]["function"]["name"] == function_name:
                    matching_tool = tool_info
                    break
            
            if not matching_tool:
                tool_results.append(f"Error: Tool {function_name} not found in any server")
                continue
                
            # Parse tool call arguments
            arguments = (
                json.loads(tool_call["function"]["arguments"])
                if isinstance(tool_call["function"]["arguments"], str)
                else tool_call["function"]["arguments"]
            )
            tool_call_id = tool_call["id"]
            
            # Create a step for the tool call
            step = None
            if msg:
                step = cl.Step(
                    name=f"Tool: {function_name} on {matching_tool['server_name']}",
                    type="tool",
                    parent_id=msg.id
                )
                await step.send()

            # Call the tool with the arguments
            try:
                tool_result = await matching_tool["callable"](**arguments)
                tool_results.append(f"Call to {function_name} ({matching_tool['server_name']}): {tool_result[:100]}...")
                
                # Update the step with the result
                if step:
                    step.content = f"Server: {matching_tool['server_name']}\nArguments: {json.dumps(arguments)}\nResult: {tool_result}"
                    await step.update()
            except Exception as e:
                error = f"Error calling tool {function_name} on {matching_tool['server_name']}: {str(e)}"
                tool_results.append(error)
                if step:
                    step.content = error
                    step.status = "error"
                    await step.update()

            # Add the tool result to the messages list
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result,
            })

        if msg:
            combined_tools = "\n".join(tool_results)
            msg.content = f"Processing tools:\n{combined_tools}"
            await msg.update()

        # Query LLM again with updated messages (including tool results)
        try:
            final_response = await model.generate_response(
                messages=messages
            )

            # Handle different response formats for the final response
            if hasattr(final_response, 'choices'):
                final_content = final_response.choices[0].message.content
            else:
                final_content = final_response.get("content", "")

            # Append the assistant's response to messages
            messages.append({"role": "assistant", "content": final_content})

            # Return the LLM response and updated messages
            return final_content, messages
        except Exception as e:
            error_msg = f"Error during second model query: {e}"
            if msg:
                msg.content = error_msg
                await msg.update()
            return error_msg, messages
    else:
        # If there were no tool calls, just return the content from the first response
        response_content = message_dict["content"]
        return response_content, messages

@cl.action_callback("view_activity")
async def show_activity(action):
    """Show recent document activity from all document servers"""
    # Show a loading message
    msg = cl.Message(content="Loading recent document activity from all servers...")
    await msg.send()
    
    try:
        activity_data_all = []
        
        # Loop through all connected servers
        for server_name, client in server_manager.servers.items():
            # Check if this server has the get_recent_activity tool
            if "get_recent_activity" in client.tools:
                try:
                    # Call the get_recent_activity tool for this server
                    activity_tool = client.tools["get_recent_activity"]["callable"]
                    server_activity = await activity_tool(limit=10)  # Get last 10 activities
                    activity_data_all.append(f"## Activities from {server_name}:\n\n{server_activity}")
                except Exception as e:
                    activity_data_all.append(f"## Error getting activities from {server_name}:\n\n{str(e)}")
        
        if activity_data_all:
            # Join all activity data with separators
            all_activities = "\n\n---\n\n".join(activity_data_all)
            msg.content = all_activities
        else:
            msg.content = "No activity data available from any servers."
            
        await msg.update()
        
        # Add a refresh button
        refresh_action = cl.Action(
            name="refresh_activity",
            label="🔄 Refresh Activity",
            description="Refreshes the document activity data from all servers",
            type="action",
            payload={}
        )
        await cl.Message(content="", actions=[refresh_action]).send()
        
    except Exception as e:
        msg.content = f"Error loading activity: {str(e)}"
        await msg.update()


@cl.action_callback("refresh_activity")
async def refresh_activity(action):
    """Refresh the activity data from all servers"""
    await show_activity(action)


@cl.on_message
async def on_message(message: cl.Message):
    global conversation_messages, server_manager, model

    # Check if model is initialized
    if not model:
        await cl.Message(content="Model not initialized. Please restart the chat.").send()
        return

    # Check for special commands
    if message.content.startswith("/display "):
        # Extract document ID and optional server name from command
        parts = message.content.replace("/display ", "").strip().split(" on ", 1)
        doc_id = parts[0]
        server_name = parts[1] if len(parts) > 1 else None
        await display_document(doc_id, server_name)
        return
    
    # Check if there are files attached to the message
    if message.elements and any(isinstance(elem, cl.File) for elem in message.elements):
        # Handle file upload
        for elem in message.elements:
            if isinstance(elem, cl.File):
                await process_file_upload(elem, message)
        return
    
    # If no files, proceed with regular message handling
    response_message = cl.Message(content="")
    await response_message.send()
    
    try:
        # Run the agent loop with the user's message and all tools from all servers
        response_content, conversation_messages = await agent_loop(
            message.content, 
            server_manager.all_tools, 
            conversation_messages,
            response_message
        )
        
        # Update the message with the final response
        response_message.content = response_content
        await response_message.update()
         
        # If the response contains document IDs in the format `/display <uuid>`, 
        # add buttons to make them clickable
        if "/display " in response_content:
            # Process the response to find document IDs
            import re
            # Match either "/display <id>" or "/display <id> on <server>"
            doc_matches = re.findall(r'/display ([a-zA-Z0-9-]+)(?:\s+on\s+([a-zA-Z0-9 ]+))?', response_content)
            
            # Create unique actions for each document
            for i, match in enumerate(set(doc_matches)):
                doc_id = match[0]
                server_name = match[1] if len(match) > 1 and match[1] else None
                
                # Create display text based on whether server was specified
                display_text = f"📄 View Document: {doc_id[:8]}..."
                if server_name:
                    display_text += f" on {server_name}"
                
                # Create an action for this specific document
                doc_action = cl.Action(
                    name=f"view_doc_{i}_{doc_id[:8]}",  # Make name unique
                    label=display_text,
                    description=f"View document with ID: {doc_id}",
                    type="action",
                    payload={"doc_id": doc_id, "server_name": server_name}
                )
                
                # Send action with empty message
                await cl.Message(content="", actions=[doc_action]).send()
                
                # Register a callback for this action
                @cl.action_callback(f"view_doc_{i}_{doc_id[:8]}")
                async def on_doc_action(action, doc_id=doc_id, server_name=server_name):
                    await display_document(doc_id, server_name)

    except Exception as e:
        response_message.content = f"Error: {str(e)}"
        await response_message.update()


async def display_document(doc_id, server_name=None):
    """Display a document's full content from a specific server or search all servers
    Shows both original and markdown versions in tabs when available"""
    # Show a loading message
    server_text = f" from {server_name}" if server_name else " (searching all servers)"
    msg = cl.Message(content=f"Loading document {doc_id}{server_text}...")
    await msg.send()
    
    try:
        # If server name is specified, use that server's display_document tool
        if server_name and server_name in server_manager.servers:
            client = server_manager.servers[server_name]
            if "display_document" in client.tools:
                display_tool = client.tools["display_document"]["callable"]
                document_data = await display_tool(doc_id=doc_id, include_original=True)
                
                # Parse the returned data - expecting JSON containing both versions
                try:
                    doc_data = json.loads(document_data)
                    await display_document_with_tabs(doc_id, doc_data, server_name, msg)
                    return
                except json.JSONDecodeError:
                    # If not JSON, just display as before
                    msg.content = f"## Document from {server_name}:\n\n{document_data}"
                    await msg.update()
                    return
            else:
                msg.content = f"Server {server_name} doesn't have a display_document tool"
                await msg.update()
                return
                
        # If no server specified or server not found, search all servers
        found = False
        for server_name, client in server_manager.servers.items():
            if "display_document" not in client.tools:
                continue
                
            try:
                display_tool = client.tools["display_document"]["callable"]
                document_data = await display_tool(doc_id=doc_id, include_original=True)
                
                # Try to parse as JSON to see if it contains both versions
                try:
                    doc_data = json.loads(document_data)
                    await display_document_with_tabs(doc_id, doc_data, server_name, msg)
                    found = True
                    break
                except json.JSONDecodeError:
                    # If not JSON, display as before
                    msg.content = f"## Document found on {server_name}:\n\n{document_data}"
                    found = True
                    await msg.update()
                    break
            except Exception as e:
                # Document not found on this server, continue to next
                pass
        
        if not found:
            msg.content = f"Document {doc_id} not found on any server"
            await msg.update()
            
        # Add back action
        back_action = cl.Action(
            name="back_to_activity",
            label="⬅️ Back to Activity",
            description="Return to activity view",
            type="action",
            payload={}
        )
        await cl.Message(content="", actions=[back_action]).send()
        
    except Exception as e:
        msg.content = f"Error displaying document: {str(e)}"
        await msg.update()

async def process_file_upload(file: cl.File, message: cl.Message):
    """Handle file uploads and process them using the ingest_document tool from available servers"""
    global server_manager, conversation_messages
    
    # Check if we have any servers with document ingestion capability
    ingest_servers = {}
    for server_name, client in server_manager.servers.items():
        if "ingest_document" in client.tools:
            ingest_servers[server_name] = client.tools["ingest_document"]["callable"]
    
    if not ingest_servers:
        await cl.Message(content="No document ingestion servers are available. Please try again later.").send()
        return
    
    # If we have multiple servers with ingestion capability, let the user choose
    server_name_to_use = None
    if len(ingest_servers) > 1:
        # Create a message asking the user to select a server
        msg = cl.Message(content=f"Multiple document servers available. Where would you like to upload '{file.name}'?")
        await msg.send()
        
        # Create an action for each server
        for server_name in ingest_servers.keys():
            server_action = cl.Action(
                name=f"upload_to_{server_name.replace(' ', '_')}",
                label=f"Upload to {server_name}",
                description=f"Upload the document to {server_name}",
                type="action",
                payload={"file_id": file.id, "server_name": server_name}
            )
            await cl.Message(content="", actions=[server_action]).send()
            
            # Register a callback for this action
            @cl.action_callback(f"upload_to_{server_name.replace(' ', '_')}")
            async def on_server_select(action, server_name=server_name):
                nonlocal server_name_to_use
                server_name_to_use = action.payload["server_name"]
                await process_upload_to_server(file, server_name_to_use)
        
        return
    else:
        # Only one server available, use it directly
        server_name_to_use = list(ingest_servers.keys())[0]
        await process_upload_to_server(file, server_name_to_use)

async def process_upload_to_server(file: cl.File, server_name: str):
    """Process a file upload to a specific server with Markitdown conversion
    Stores both original and converted markdown content in MongoDB"""
    global server_manager, conversation_messages
    
    # Create a message to update with progress
    process_message = cl.Message(content=f"Processing uploaded file: {file.name} on {server_name}")
    await process_message.send()
    
    try:
        # Get the client for the selected server
        client = server_manager.servers[server_name]
        try:
            # Get file content
            content = await get_file_content(file)
            if not content:
                raise ValueError("Unable to extract file content")
        except Exception as e:
            process_message.content = f"Error reading file: {str(e)}"
            await process_message.update()
            return
            
        # Convert content to string if needed
        if isinstance(content, bytes):
            try:
                content_str = content.decode('utf-8')
            except UnicodeDecodeError:
                # If UTF-8 decoding fails, try another common encoding
                try:
                    content_str = content.decode('latin-1')
                except:
                    content_str = str(content)  # Last resort
        else:
            content_str = str(content)
        
        # Keep original content for storage
        original_content = content_str
        
        # Get file extension
        file_extension = os.path.splitext(file.name)[1].lower().replace('.', '')
        
        # Create a step for Markitdown conversion
        conversion_step = cl.Step(
            name="Converting document to Markdown",
            type="tool",
            parent_id=process_message.id
        )
        await conversion_step.send()
        
        try:
            # Get Markitdown server client
            markitdown_client = server_manager.servers.get("Markitdown Server")
            if markitdown_client and "convert_to_markdown" in markitdown_client.tools:
                convert_tool = markitdown_client.tools["convert_to_markdown"]["callable"]
                
                # Convert document to markdown
                markdown_content = await convert_tool(
                    content=content_str,
                    source_format=file_extension or "txt"
                )
                
                conversion_step.content = "Successfully converted document to Markdown"
                await conversion_step.update()
            else:
                markdown_content = content_str
                conversion_step.content = "Markitdown server not available, using original content"
                await conversion_step.update()
        except Exception as e:
            conversion_step.content = f"Error converting to Markdown: {str(e)}"
            conversion_step.status = "error"
            await conversion_step.update()
            markdown_content = content_str
        
        # Create a step for document ingestion
        ingestion_step = cl.Step(
            name=f"Ingesting document: {file.name}",
            type="tool",
            parent_id=process_message.id
        )
        await ingestion_step.send()
        
        # Get the ingest_document tool
        if "ingest_document" not in client.tools:
            raise ValueError(f"Server {server_name} doesn't have an ingest_document tool")
            
        ingest_tool = client.tools["ingest_document"]["callable"]
        
        # Prepare metadata
        metadata = {
            "filename": file.name,
            "original_size": len(original_content),
            "converted_size": len(markdown_content),
            "original_format": file_extension or "txt",
            "uploaded_at": datetime.now().isoformat(),
            "uploaded_by": "user",
            "server": server_name,
            "conversion_status": "converted" if markdown_content != original_content else "original",
            "has_original": True,  # Flag to indicate original content is stored
            "has_markdown": True,  # Flag to indicate markdown content is stored
        }
        
        # Generate summary
        summary = generate_document_summary(file.name, markdown_content)
        
        # Convert metadata to JSON string
        entities_json = json.dumps(metadata)
        
        # Store both original content and markdown content
        # We'll modify the ingest_document function to handle both versions
        doc_id = await ingest_tool(
            content=markdown_content,
            original_content=original_content,  # Add original content
            doc_type="md",  # Primary storage as markdown
            original_type=file_extension or "txt",  # Original file type
            entities=entities_json,
            summary=summary
        )
        
        # Update ingestion step
        ingestion_step.content = (
            f"Successfully stored document in MongoDB.\n"
            f"Document ID: {doc_id}\n"
            f"Both original and Markdown versions stored.\n"
            f"Conversion: {metadata['conversion_status']}"
        )
        await ingestion_step.update()
        
        # Send final confirmation message
        await cl.Message(
            content=(
                f"✅ Document '{file.name}' has been processed and stored.\n\n"
                f"📝 **Details:**\n"
                f"- **Document ID:** {doc_id}\n"
                f"- **Original Format:** {file_extension or 'txt'}\n"
                f"- **Stored Format:** Both original and Markdown\n"
                f"- **Original Size:** {len(original_content)} characters\n"
                f"- **Converted Size:** {len(markdown_content)} characters\n\n"
                f"You can retrieve this document using:\n"
                f"`/display {doc_id}`"
            )
        ).send()
        
        # Add view document action
        view_action = cl.Action(
            name=f"view_doc_{doc_id[:8]}",
            label="📄 View Document",
            description=f"View document with ID: {doc_id}",
            type="action",
            payload={"doc_id": doc_id, "server_name": server_name}
        )
        await cl.Message(content="", actions=[view_action]).send()
        
    except Exception as e:
        error_message = f"Error processing file: {str(e)}"
        print(f"Detailed error: {type(e).__name__}: {str(e)}")
        
        if 'conversion_step' in locals() and not conversion_step.status:
            conversion_step.content = error_message
            conversion_step.status = "error"
            await conversion_step.update()
        
        if 'ingestion_step' in locals() and not ingestion_step.status:
            ingestion_step.content = error_message
            ingestion_step.status = "error"
            await ingestion_step.update()
        
        process_message.content = error_message
        await process_message.update()


# Helper functions

async def get_file_content(file: cl.File) -> str | bytes | None:
    """Get file content using various methods"""
    try:
        # Try to access content directly as an attribute
        if hasattr(file, 'content') and file.content is not None:
            return file.content
        
        # Get content through method if available
        if hasattr(file, 'get_content'):
            try:
                return await file.get_content()
            except Exception as e:
                print(f"get_content() method failed: {e}")
        
        # Try the path approach
        try:
            if hasattr(file, 'path') and file.path:
                with open(file.path, 'rb') as f:
                    return f.read()
        except Exception as e:
            print(f"Path approach failed: {e}")
            
        # Try to get file path and read from it
        if hasattr(file, 'get_path'):
            try:
                file_path = await file.get_path()
                with open(file_path, 'rb') as f:
                    return f.read()
            except Exception as e:
                print(f"get_path() method failed: {e}")
        
        # Save file to temp location
        try:
            temp_path = f"/tmp/{file.name}"
            if hasattr(file, 'save'):
                await file.save(temp_path)
                with open(temp_path, 'rb') as f:
                    content = f.read()
                os.remove(temp_path)  # Clean up
                return content
        except Exception as e:
            print(f"Save approach failed: {e}")
            
        # Try other attributes
        if hasattr(file, 'bytes') and file.bytes is not None:
            return file.bytes
            
        # Try to read content as a stream
        if hasattr(file, 'read'):
            try:
                return await file.read()
            except Exception as e:
                print(f"read() method failed: {e}")
                
        # Try to access file data through the Element
        if hasattr(file, 'data') and file.data is not None:
            return file.data
        
        # Get file content using Element's name
        if isinstance(file, cl.Element):
            try:
                for elem in cl.user_session.get("files", []):
                    if elem.name == file.name:
                        if hasattr(elem, 'content') and elem.content:
                            return elem.content
            except Exception as e:
                print(f"Element approach failed: {e}")
        
        # Try accessing base64 content and decode
        if hasattr(file, 'base64') and file.base64:
            return base64.b64decode(file.base64)
            
        raise ValueError("Could not read file content - no suitable method found")
    except Exception as e:
        print(f"Error reading file content: {str(e)}")
        return None

def generate_document_summary(filename: str, content: str, max_length: int = 200) -> str:
    """Generate a simple summary of the document"""
    # Get first few lines
    lines = content.split('\n')[:5]
    preview = '\n'.join(lines)
    
    if len(content) > len(preview):
        preview += "..."
        
    return f"Document: {filename} - {len(content)} characters\nPreview:\n{preview[:max_length]}"

@cl.action_callback("view_doc_.*")
async def on_view_document(action):
    """Handle document view action"""
    doc_id = action.payload.get("doc_id")
    server_name = action.payload.get("server_name")
    if doc_id:
        await display_document(doc_id, server_name)

def get_source_format(filename: str) -> str:
    """Determine the source format from filename"""
    extension = os.path.splitext(filename)[1].lower().replace('.', '')
    format_mapping = {
        'doc': 'docx',
        'docx': 'docx',
        'pdf': 'pdf',
        'txt': 'txt',
        'rtf': 'rtf',
        'odt': 'odt',
        'html': 'html',
        'htm': 'html',
        'md': 'md',
        'markdown': 'md',
    }
    return format_mapping.get(extension, 'txt')

async def display_document_with_tabs(doc_id, doc_data, server_name, msg):
    """Display both original and markdown versions of a document using tabs"""
    try:
        # Extract document versions and metadata
        markdown_content = doc_data.get("markdown_content", "")
        original_content = doc_data.get("original_content", "")
        metadata = doc_data.get("metadata", {})
        
        # Create tabs for display
        tabs = []
        
        # Preview/Markdown tab
        if markdown_content:
            markdown_tab = cl.Tab(name="markdown", label="Markdown")
            markdown_element = cl.Text(name="markdown_content", content=markdown_content)
            markdown_tab.elements = [markdown_element]
            tabs.append(markdown_tab)
        
        # Original document tab
        if original_content:
            original_format = metadata.get("original_format", "txt")
            original_tab = cl.Tab(name="original", label=f"Original ({original_format.upper()})")
            original_element = cl.Text(name="original_content", content=original_content)
            original_tab.elements = [original_element]
            tabs.append(original_tab)
        
        # Metadata tab
        if metadata:
            metadata_tab = cl.Tab(name="metadata", label="Metadata")
            metadata_content = "## Document Metadata\n\n"
            for key, value in metadata.items():
                metadata_content += f"**{key}**: {value}\n"
            metadata_element = cl.Text(name="metadata_content", content=metadata_content)
            metadata_tab.elements = [metadata_element]
            tabs.append(metadata_tab)
        
        # Create a tabbed interface
        tab_container = cl.TabContainer(tabs=tabs)
        
        # Update message with content and tabs
        msg.content = f"## Document ID: {doc_id} (from {server_name})"
        msg.elements = [tab_container]
        await msg.update()
        
    except Exception as e:
        msg.content = f"Error displaying document with tabs: {str(e)}"
        await msg.update()

@cl.on_chat_end
async def on_chat_end():
    global mcp_client, model
    if model:
        await model.cleanup()
    # Clean up MCP client when the chat ends
    if mcp_client:
        try:
            await mcp_client.__aexit__(None, None, None)
            mcp_client = None
        except Exception as e:
            print(f"Error closing MCP client: {e}")