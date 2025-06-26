import json
import os
import logging
from typing import Any, List, Dict, Union, Optional
import base64
from pathlib import Path
import asyncio
import chainlit as cl
from openai import AsyncOpenAI
from datetime import datetime
from models.openai_model import OpenAIModel
from models.ollama_model import OllamaModel
from MCPClient import MCPClient
from MCPServerManager import MCPServerManager
import sys
# Add LangSmith imports
from langsmith import Client as LangSmithClient
from langsmith.run_trees import RunTree

# Configure more visible logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG for more verbose logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.info("Client initialized with enhanced logging")

# Global variables
model = None
server_manager = MCPServerManager()
conversation_messages = None
originals_dir = "originalfiles"
markdown_dir = "markdown_files"

# Initialize LangSmith client
langsmith_client = LangSmithClient(
    api_key=os.getenv("LANGSMITH_API_KEY","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"),
    api_url=os.getenv("LANGSMITH_API_URL", "https://api.smith.langchain.com")
)
project_name = os.getenv("LANGSMITH_PROJECT", "LangGraph-MCP-Agents")


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

@cl.on_chat_start
async def on_chat_start():
    global server_manager, conversation_messages, model
    
    # Initialize the model
    try:
        model = OpenAIModel(model_id="gpt-4o-mini")  # or OllamaModel(model_name="llama2")
        await model.initialize()
    except Exception as e:
        error_msg = f"Error initializing model: {str(e)}"
        logger.error(error_msg)
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
            "name": "Markitdown Server",
            "command": "docker",
            "args": [
                "run",
                "--rm",
                "-i",
                 "-v",
                f"{os.path.abspath(originals_dir)}:/documents",
                "-v",
                "mcp-test:/mcp",
                "mcp-markdown",
            ],
            "env": None,
        },
        # {
        #     "name": "Pandoc Server",
        #     "command": "docker",
        #     "args": [
        #         "run",
        #         "--rm",
        #         "-i",
        #         "-v",
        #         "data:/app/data",
        #         "mcp-pandoc",
        #     ],
        #     "env": None,
        # },
        
    ]
    
    # Start MCP clients for each server - use sequential connection to avoid race conditions
    connected_servers = []
    failed_servers = []
    
    # Connect to servers sequentially to further prevent race conditions
    for server_config in server_configs:
        server_name = server_config["name"]
        welcome_msg.content = f"Connecting to {server_name}..."
        await welcome_msg.update()
        
        try:
            success = await server_manager.add_server(server_config)
            if success:
                # Verify server tools after connection
                server = server_manager.get_server(server_name)
                if server and server.tools:
                    connected_servers.append(server_name)
                    logger.info(f"Successfully connected to {server_name} with tools: {list(server.tools.keys())}")
                else:
                    failed_servers.append(server_name)
                    logger.error(f"Server {server_name} connected but has no tools")
            else:
                failed_servers.append(server_name)
                logger.error(f"Failed to connect to {server_name}")
        except Exception as e:
            failed_servers.append(server_name)
            logger.error(f"Error connecting to {server_name}: {e}")
    
    # Prepare message about connected servers
    if connected_servers:
        server_list = ", ".join(connected_servers)
        tools_count = len(server_manager.all_tools)
        
        # Log available tools
        logger.info(f"Available tools: {list(server_manager.all_tools.keys())}")
         # Check if Pandoc Server is connected
        if "Pandoc Server" in connected_servers:
            pandoc_server = server_manager.get_server("Pandoc Server")
            if "list_formats" in pandoc_server.tools:
                try:
                    # Get supported formats
                    list_formats_tool = pandoc_server.tools["list_formats"]["callable"]
                    formats_result = await list_formats_tool()
                    formats_info = json.loads(formats_result)
                    logger.info(f"Pandoc supports {len(formats_info.get('input_formats', []))} input formats and {len(formats_info.get('output_formats', []))} output formats")
                except Exception as e:
                    logger.error(f"Error getting Pandoc formats: {e}")
        
        success_message = (
            f"Connected to {len(connected_servers)} MCP servers: {server_list}\n"
            f"Found {tools_count} tools across all servers\n\n"
            "You can upload documents using the file upload button."
        )    
    else:
        success_message = "Failed to connect to any MCP servers. Please check configurations and try again."
    
    # Report on failed servers
    if failed_servers:
        failed_list = ", ".join(failed_servers)
        success_message += f"\n\nFailed to connect to: {failed_list}"
    
    # Update the welcome message with results
    welcome_msg.content = success_message
    await welcome_msg.update()

async def agent_loop(query: str, tools: Dict, messages: List[Dict] = None, msg=None) -> tuple[str, List[Dict]]:
    """
    Process user query using the LLM and available tools
    """
    if messages is None:
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    tools="\n- ".join([
                        f"{t['name']} ({t['server_name']}): {t['schema']['function']['description']}"
                        for t in tools.values()
                    ])
                )
            }
        ]

    # Add user query to messages
    messages.append({"role": "user", "content": query})

    # Show thinking indicator
    if msg:
        msg.content = "Thinking..."
        await msg.update()

    try:
        # Get initial response from model
        response = await model.generate_response(
            messages=messages,
            tools=[t["schema"] for t in tools.values()] if tools else None
        )

        # Handle different response formats
        if isinstance(response, dict):
            # Handle dictionary format
            content = response.get("content", "")
            tool_calls = response.get("tool_calls", [])
            
            # Add assistant's response to messages
            messages.append({
                "role": "assistant",
                "content": content or "",
                **({"tool_calls": tool_calls} if tool_calls else {})
            })

            # Process tool calls if present
            if tool_calls:
                if msg:
                    msg.content = "Calling tools..."
                    await msg.update()

                tool_results = []

                # Process each tool call
                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    arguments = json.loads(tool_call["function"]["arguments"])
                    tool_call_id = tool_call["id"]

                    # Find matching tool
                    matching_tool = None
                    for tool in tools.values():
                        if tool["schema"]["function"]["name"] == function_name:
                            matching_tool = tool
                            break

                    if not matching_tool:
                        error = f"Tool not found: {function_name}"
                        tool_results.append(error)
                        continue

                    # Create step for tool call
                    step = None
                    if msg:
                        step = cl.Step(
                            name=f"Tool: {function_name}",
                            type="tool",
                            parent_id=msg.id
                        )
                        await step.send()

                    try:
                        # Call the tool
                        result = await matching_tool["callable"](**arguments)
                        result_preview = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
                        tool_results.append(f"Tool {function_name}: {result_preview}")

                        if step:
                            step.content = f"Result: {result}"
                            await step.update()

                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": str(result)
                        })

                    except Exception as e:
                        error = f"Error calling {function_name}: {str(e)}"
                        tool_results.append(error)
                        if step:
                            step.content = error
                            step.status = "error"
                            await step.update()

                if msg:
                    msg.content = "Processing tool results..."
                    await msg.update()

                # Get final response from model
                final_response = await model.generate_response(messages=messages)
                if isinstance(final_response, dict):
                    final_content = final_response.get("content", "")
                else:
                    final_content = str(final_response)
                    
                messages.append({"role": "assistant", "content": final_content})
                return final_content, messages

            else:
                # No tool calls, return content directly
                return content, messages

        else:
            # Handle non-dictionary response
            response_content = str(response)
            messages.append({"role": "assistant", "content": response_content})
            return response_content, messages

    except Exception as e:
        error_msg = f"Error in agent loop: {str(e)}"
        logger.error(error_msg, exc_info=True)
        if msg:
            msg.content = f"❌ {error_msg}"
            await msg.update()
        return f"Error: {str(e)}", messages

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

async def process_file_upload(file: cl.File, msg: cl.Message):
    """
    Handle file upload processing with dual storage:
    1. Store original file in its original format
    2. Convert and store a markdown version of the file
    """
    
    logger.info(f"==== STARTING FILE UPLOAD PROCESS: {file.name} ====")
    msg.content = f"Processing file: {file.name}"
    await msg.update()
    
    # Check server manager and document server
    if not server_manager:
        msg.content = "Error: Server manager is not initialized"
        await msg.update()
        return

    doc_server = server_manager.get_server("Document Server")
    if not doc_server:
        msg.content = "Error: Document server is not available"
        await msg.update()
        return

    # Verify document server has the required tool
    if "ingest_document" not in doc_server.tools:
        msg.content = "Error: Document server does not have ingest_document tool"
        await msg.update()
        return
    
    # Ensure directories exist
    os.makedirs(originals_dir, exist_ok=True)
    os.makedirs(markdown_dir, exist_ok=True)
    logger.info(f"Ensuring storage directories exist: {originals_dir}, {markdown_dir}")
    
    # Create a step for file processing
    process_step = cl.Step(
        name=f"Processing file: {file.name}",
        type="tool", 
        parent_id=msg.id
    )
    await process_step.send()
    
    try:
        # Extract file content with better error handling
        logger.info(f"Attempting to extract content from file: {file.name}")
        binary_content = await get_file_content(file)
        
        if binary_content is None or (isinstance(binary_content, (bytes, str)) and len(binary_content) == 0):
            error_msg = f"Unable to extract content from file: {file.name}. File may be empty or inaccessible."
            logger.error(error_msg)
            process_step.content = error_msg
            process_step.status = "error"
            await process_step.update()
            return
            
        logger.info(f"Successfully extracted {len(binary_content)} bytes from {file.name}")
        process_step.content = f"Successfully read file content ({len(binary_content)} bytes)"
        await process_step.update()
        
        # Determine if binary content needs base64 encoding
        file_extension = os.path.splitext(file.name)[1].lower().replace('.', '')
        binary_extensions = ['pdf', 'docx', 'xlsx', 'pptx', 'odt', 'epub', 'jpg', 'png', 'gif']
        is_binary = file_extension in binary_extensions
        if is_binary and isinstance(binary_content, str):
            try:
                binary_content = binary_content.encode('utf-8')
            except:
                # If encoding fails, it might already be a base64 string
                try:
                    binary_content = base64.b64decode(binary_content)
                except:
                    pass  # Keep as is if decoding fails

        if is_binary:
            logger.info(f"Detected binary file type: {file.name}")
            # For binary files, encode as base64 for transmission
            if isinstance(binary_content, bytes):
                content_str = base64.b64encode(binary_content).decode('ascii')
                logger.info(f"Encoded binary file as base64 for transport (length: {len(content_str)})")
            else:
                # If somehow already a string, keep as is
                content_str = binary_content
                logger.info("Binary file content already in string format")
        else:
            # Convert content to string if needed
            logger.info("Handling as text file")
            if isinstance(binary_content, bytes):
                try:
                    content_str = binary_content.decode('utf-8')
                    logger.info("Successfully decoded content as UTF-8")
                except UnicodeDecodeError:
                    try:
                        content_str = binary_content.decode('latin-1')
                        logger.info("Successfully decoded content as Latin-1")
                    except:
                        # Base64 encode binary content if text decoding fails
                        content_str = base64.b64encode(binary_content).decode('ascii')
                        logger.info("Used base64 encoding for binary content after text decode failed")
                        is_binary = True
            else:
                content_str = str(binary_content)
                logger.info("Content was already string-like")
        
        logger.info("About to assign original_content")
        logger.info(f"Content prepared for transmission (is_binary: {is_binary}, length: {len(content_str)})")
        original_content = content_str

        logger.info(f"original_content assigned with length: {len(original_content)}")

        logger.info("About to extract file extension")
        file_extension = os.path.splitext(file.name)[1].lower().replace('.', '')
        logger.info(f"File extension extracted: {file_extension}")

        logger.info("About to create conversion step")
        # Create conversion step
        conversion_step = cl.Step(
            name="Converting document to Markdown",
            type="tool",
            parent_id=msg.id
        )
        logger.info("Conversion step created, about to send")
        await conversion_step.send()
        logger.info("Conversion step sent")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Get base filename and extension separately
        base_filename = os.path.splitext(file.name)[0]
        file_ext = os.path.splitext(file.name)[1]  # This will include the dot, e.g., '.pdf'

        # Make the base filename safe by replacing non-alphanumeric chars with underscores
        safe_base_filename = "".join([c if c.isalnum() else "_" for c in base_filename])

        # Save original file with timestamp and proper extension
        original_filepath = os.path.join(originals_dir, f"{timestamp}_{safe_base_filename}{file_ext}")

        # Write original file content
        # For the original file:
        with open(original_filepath, 'wb' if is_binary else 'w', encoding='utf-8' if not is_binary else None) as f:
            if is_binary:
                if isinstance(binary_content, bytes):
                    f.write(binary_content)
                else:
                    # If it's a string but should be binary, try to decode base64
                    try:
                        if binary_content.startswith("base64:"):
                            binary_content = binary_content[7:]  # Remove prefix
                        f.write(base64.b64decode(binary_content))
                    except:
                        f.write(binary_content.encode('utf-8'))
            else:
                # For text files - explicitly use UTF-8 encoding
                if isinstance(binary_content, bytes):
                    f.write(binary_content.decode('utf-8', errors='replace'))
                else:
                    f.write(binary_content)

        logger.info(f"Original file saved locally at: {original_filepath}")

        # Initialize markdown_content as None - THIS IS THE FIX
        markdown_content = None
        pandoc_server = None  # Initialize this variable since it was referenced
        markitdown_server = None  # Initialize this for clarity

        # Try Markitdown server for conversion
        markitdown_server = server_manager.get_server("Markitdown Server")
        if markitdown_server and "convert_to_markdown" in markitdown_server.tools:
            logger.info("Using Markitdown server for conversion")
            convert_tool = markitdown_server.tools["convert_to_markdown"]["callable"]
            
            try:
                abs_path = os.path.abspath(original_filepath)
                filename = os.path.basename(original_filepath)
                logger.info(f"Absolute file path: {abs_path}")
        
                # Format the path properly for the URI
                # Replace backslashes with forward slashes for URI compatibility
                uri_path = abs_path.replace('\\', '/')
                        
                # Add file:// scheme - for Windows paths with drive letter
                if os.name == 'nt' and uri_path[1:3] == ':/':  # Windows path with drive letter
                    file_uri = f"file:////{uri_path}"
                else:  # Unix path
                    file_uri = f"file:///{uri_path}"
                    # Convert document to markdown
                logger.info(f"File path converted to URI: {file_uri}")
                file_path_in_container = f"file:/documents/{os.path.basename(original_filepath)}"
                # Convert document to markdown
                logger.info(f"Converting from format: {file_extension} using Markitdown with URI: {file_uri}")
                markdown_content = await convert_tool(uri=file_path_in_container)
                logger.info(f"Markitdown conversion successful, content length: {len(markdown_content)}")
                conversion_step.content = "Successfully converted document to Markdown using Markitdown"

                markdown_filepath = os.path.join(markdown_dir, f"{timestamp}_{safe_base_filename}.md")
                with open(markdown_filepath, 'w', encoding='utf-8') as f:
                        f.write(markdown_content)
                    
                logger.info(f"Markdown version saved locally at: {markdown_filepath}")

                await conversion_step.update()
            except Exception as e:
                logger.error(f"Error converting with Markitdown: {str(e)}", exc_info=True)
                conversion_step.content = f"Markitdown conversion failed: {str(e)}"
                conversion_step.status = "error"
                await conversion_step.update()
                # If conversion failed, set markdown_content to original content for text files
                if not is_binary:
                    markdown_content = content_str
                    logger.info("Using original content as markdown for text files")
        else:
            logger.warning("No conversion servers available")
            conversion_step.content = "No conversion servers available, using original content"
            await conversion_step.update()
            
            # For text files, use the original content as markdown
            if not is_binary:
                markdown_content = content_str
                logger.info("Using original content as markdown for text files")
                markdown_filepath = os.path.join(markdown_dir, f"{timestamp}_{safe_base_filename}.md")
                with open(markdown_filepath, 'w', encoding='utf-8') as f:
                    f.write(content_str)
                logger.info(f"Original content saved as markdown at: {markdown_filepath}")
        
        # If markdown_content is still None, create a basic placeholder
        if markdown_content is None:
            markdown_content = f"# {file.name}\n\nThis document was uploaded but could not be converted to markdown format."
            logger.info("Created placeholder markdown content")
            markdown_filepath = os.path.join(markdown_dir, f"{timestamp}_{safe_base_filename}.md")
            with open(markdown_filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            logger.info(f"Placeholder markdown saved at: {markdown_filepath}")

        ingest_step = cl.Step(
            name=f"Ingesting document: {file.name}",
            type="tool", 
            parent_id=msg.id
        )
        await ingest_step.send()
        
        # Actually call the ingest_document tool from the Document Server
        try:
            logger.info("Getting ingest_document tool")
            ingest_tool = doc_server.tools["ingest_document"]["callable"]
            
            # Prepare metadata for the document
            metadata = {
                "filename": file.name,
                "original_format": file_extension or "txt",
                "uploaded_at": datetime.now().isoformat(),
                "size": len(binary_content) if isinstance(binary_content, bytes) else len(str(binary_content)),
                "original_filename": file.name,
                "storage_path": originals_dir,
                "has_markdown": True,
                "markdown_path": markdown_dir,
                "is_binary": is_binary,
                "conversion_method": "pandoc" if pandoc_server else "markitdown" if markitdown_server else "none"
            }
            logger.info(f"Prepared metadata: {metadata}")
            
            # Create a summary of the document (this could be improved with LLM-based summarization)
            summary = f"Uploaded document: {file.name}"
            
            # The document server needs to know to decode this back to binary
            if is_binary:
                # Encode binary data to base64 string with prefix
                if isinstance(binary_content, bytes):
                    original_content_transport = "base64:" + base64.b64encode(binary_content).decode('ascii')
                    logger.info("Prepared binary content with base64 prefix for transport")
                else:
                    # If already a string (which shouldn't happen for binary), prefix it
                    original_content_transport = "base64:" + binary_content if not binary_content.startswith("base64:") else binary_content
                    logger.info("Using existing string content with base64 prefix")
            else:
                # For text files, just use the content string directly
                if isinstance(binary_content, bytes):
                    try:
                        original_content_transport = binary_content.decode('utf-8')
                    except:
                        original_content_transport = binary_content.decode('latin-1', errors='replace')
                else:
                    original_content_transport = binary_content
                logger.info("Using text content for transport")
            
            # Call the ingest_document tool with file details
            result = await ingest_tool(
                content=markdown_content,  # Store the markdown version as primary content
                doc_type=file_extension or "txt",
                original_content=original_content_transport, 
                entities=json.dumps(metadata),
                summary=summary
            )
            logger.info("ingestion tool called")
            # Update the ingestion step with the result
            ingest_step.content = f"Document ingestion result: {result}"
            await ingest_step.update()
            logger.info("ingestion step updated")
            # Update main message with success
            msg.content = f"✅ Successfully ingested document: {file.name}"
            await msg.update()
            
            # Add context to conversation about the document
            global conversation_messages
            if conversation_messages is not None:
                system_message = f"The user has uploaded a document named '{file.name}' which has been processed and is now available for reference. The document ID is: {result}"
                conversation_messages.append({"role": "system", "content": system_message})
                
        except Exception as e:
            error_msg = f"Error ingesting document: {str(e)}"
            logger.error(error_msg, exc_info=True)
            ingest_step.content = error_msg
            ingest_step.status = "error"
            await ingest_step.update()
            
            msg.content = f"❌ Error during document ingestion: {str(e)}"
            await msg.update()
            
    except Exception as e:
        error_message = f"Error processing file {file.name}: {str(e)}"
        logger.error(error_message, exc_info=True)
        
        if 'process_step' in locals():
            process_step.content = error_message
            process_step.status = "error"
            await process_step.update()
        
        msg.content = f"❌ {error_message}"
        await msg.update()
    logger.info(f"==== FILE UPLOAD PROCESS COMPLETED: {file.name} ====")

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming chat messages"""
    global server_manager, model, conversation_messages
    
    if not server_manager or not model:
        await cl.Message(content="❌ Servers not initialized. Please restart the chat.").send()
        return

    # Handle file uploads with better logging
    if message.elements and any(isinstance(elem, cl.File) for elem in message.elements):
        logger.info(f"Received message with {len(message.elements)} elements, including files")
        for element in message.elements:
            if isinstance(element, cl.File):
                logger.info(f"Processing file upload: {element.name}")
                await process_file_upload(element, message)
        return

    # Create response message
    msg = cl.Message(content="")
    await msg.send()

    try:
        # Get all available tools
        tools = server_manager.get_tools()
        
        # Initialize conversation if needed
        if conversation_messages is None:
            conversation_messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(
                        tools="\n- ".join([
                            f"{t['name']} ({t['server_name']}): {t['schema']['function']['description']}"
                            for t in tools.values()
                        ])
                    )
                }
            ]

        # Process the message
        try:
            response, updated_messages = await agent_loop(
                query=message.content,
                tools=tools,
                messages=conversation_messages,
                msg=msg
            )
            
            # Update conversation history
            conversation_messages = updated_messages
            
            # Update response message
            msg.content = response
            await msg.update()
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg, exc_info=True)
            msg.content = f"❌ {error_msg}"
            await msg.update()

    except Exception as e:
        error_msg = f"Error in message handler: {str(e)}"
        logger.error(error_msg, exc_info=True)
        msg.content = f"❌ {error_msg}"
        await msg.update()

@cl.on_chat_end
async def on_chat_end():
    global server_manager, model
    
    try:
        if model:
            await model.cleanup()
            
        if server_manager:
            await server_manager.close_all()
    except Exception as e:
        logger.error(f"Error closing resources: {e}")