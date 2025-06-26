import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

# Import MCP client modules
from MCPClient import MCPClient as Client
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def read_file_content(file_path: str) -> str:
    """Read the content of a file as string."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        # If utf-8 fails, try reading as binary and then decode
        with open(file_path, 'rb') as file:
            return file.read().decode('utf-8', errors='replace')

async def main():
    # Check if file path is provided
    if len(sys.argv) < 2:
        print("Usage: python markitdown_client.py <file_to_convert> [source_format]")
        sys.exit(1)
        
    file_path = sys.argv[1]
    
    # Determine source format from file extension or use provided format
    if len(sys.argv) >= 3:
        source_format = sys.argv[2]
    else:
        # Extract extension from file path
        extension = Path(file_path).suffix.lower().lstrip('.')
        source_format = extension if extension else "txt"
    
    print(f"Converting file: {file_path} (format: {source_format})")
    
    # Read file content
    try:
        file_content = await read_file_content(file_path)
        print(f"Read {len(file_content)} characters from file.")
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Set up MCP client
    client = Client()
    
    # Get document ID from filename (without extension)
    doc_id = Path(file_path).stem
    
    try:
        # Connect to the MCP server using stdio transport
        async with mcp.client.stdio.stdio_client() as (read_stream, write_stream):
            # Initialize the client
            await client.initialize(
                read_stream,
                write_stream,
                InitializationOptions(
                    client_name="markitdown-client",
                    client_version="0.1.0",
                    capabilities=ClientCapabilities(
                        experimental_capabilities={}
                    ),
                ),
            )
            
            # List available tools
            tools = await client.list_tools()
            print(f"Available tools: {', '.join(tool.name for tool in tools)}")
            
            # Verify that convert_to_markdown tool is available
            convert_tool = next((tool for tool in tools if tool.name == "convert_to_markdown"), None)
            if not convert_tool:
                print("Error: convert_to_markdown tool not found")
                sys.exit(1)
            
            # Call convert_to_markdown tool
            print("Calling convert_to_markdown...")
            result = await client.call_tool(
                "convert_to_markdown",
                {
                    "content": file_content,
                    "source_format": source_format,
                    "doc_id": doc_id
                }
            )
            
            # Process the result
            if result and len(result) > 0:
                markdown_content = result[0].text
                
                # Save the markdown content to a file
                output_file = f"{doc_id}.md"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                print(f"Markdown content saved to {output_file}")
                
                # Print first 100 characters of the markdown
                preview = markdown_content[:100] + "..." if len(markdown_content) > 100 else markdown_content
                print(f"Preview: {preview}")
            else:
                print("No result returned from convert_to_markdown")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print("Conversion completed successfully")

if __name__ == "__main__":
    asyncio.run(main())