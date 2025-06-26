import json
import os
from typing import Any, List, Dict, Union, Optional
import base64
from contextlib import asynccontextmanager
import asyncio
import logging
from datetime import datetime
import anyio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

class MCPClient:
    """
    A client class for interacting with the MCP (Model Control Protocol) server.
    This class manages the connection and communication with a specific MCP server.
    """

    def __init__(self, server_params: StdioServerParameters, server_name: str):
        """Initialize the MCP client with server parameters and name"""
        self.server_params = server_params
        self.server_name = server_name
        self.session = None
        self._client = None
        self.tools = {}
        self.resources = []
        self.read = None
        self.write = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        try:
            if self.session:
                await self.session.__aexit__(exc_type, exc_val, exc_tb)
                self.session = None
            if self._client:
                await self._client.__aexit__(exc_type, exc_val, exc_tb)
                self._client = None
        except Exception as e:
            print(f"Error closing MCP client for {self.server_name}: {e}")

    async def connect(self):
        """Establishes connection to MCP server"""
        try:
            self._client = stdio_client(self.server_params)
            self.read, self.write = await self._client.__aenter__()
            self.session = ClientSession(self.read, self.write)
            await self.session.__aenter__()
            await self.session.initialize()
            print(f"Successfully connected to {self.server_name}")
        except Exception as e:
            print(f"Error connecting to {self.server_name}: {str(e)}")
            if self.session:
                try:
                    await self.session.__aexit__(None, None, None)
                except Exception:
                    pass
                self.session = None
            if self._client:
                try:
                    await self._client.__aexit__(None, None, None)
                except Exception:
                    pass
                self._client = None
            raise
    async def get_available_tools(self) -> List[Any]:
        """
        Retrieve a list of available tools from the MCP server.
        """
        if not self.session:
            raise RuntimeError(f"Not connected to MCP server {self.server_name}")

        try:
            tools_response = await self.session.list_tools()
            tools_list = tools_response.tools
            
            # Create a callable for each tool and store in the tools dict
            self.tools = {
                tool.name: {
                    "name": tool.name,
                    "callable": self.call_tool(tool.name),
                    "schema": {
                        "type": "function",
                        "function": {
                            "name": f"{self.server_name.replace(' ', '_')}_{tool.name}",  # Prefix with server name to avoid conflicts
                            "description": f"[{self.server_name}] {tool.description}",
                            "parameters": tool.inputSchema,
                        },
                    },
                    "server_name": self.server_name,
                }
                for tool in tools_list
            }
            
            print(f"Found {len(self.tools)} tools on {self.server_name}: {', '.join(self.tools.keys())}")
            return self.tools
        except Exception as e:
            print(f"Error getting tools from {self.server_name}: {str(e)}")
            raise
            
    async def get_available_resources(self) -> List[Any]:
        """
        Retrieve a list of available resources from the MCP server.
        """
        if not self.session:
            raise RuntimeError(f"Not connected to MCP server {self.server_name}")
            
        resources_response = await self.session.list_resources()
        self.resources = resources_response.resources
        
        # Add server name to each resource for identification
        for resource in self.resources:
            resource.server_name = self.server_name
            
        return self.resources
    
    async def read_resource(self, uri: str) -> str:
        """
        Read a specific resource from the MCP server.
        """
        if not self.session:
            raise RuntimeError(f"Not connected to MCP server {self.server_name}")
            
        response = await self.session.read_resource(uri)
        return response.content

    def call_tool(self, tool_name: str) -> Any:
        """
        Create a callable function for a specific tool.
        """
        if not self.session:
            raise RuntimeError(f"Not connected to MCP server {self.server_name}")

        async def callable(*args, **kwargs):
            response = await self.session.call_tool(tool_name, arguments=kwargs)
            return response.content[0].text

        return callable

    def get_tools(self) -> Dict[str, Any]:
        """
        Get the dictionary of available tools
        """
        return self.tools

    async def close(self):
        """
        Close the connection to the MCP server
        """
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
                self.session = None
                
            if self._client:
                await self._client.__aexit__(None, None, None)
                self._client = None
                
            self.connected = False
            logger.info(f"Closed connection to {self.server_name}")
            
        except Exception as e:
            logger.error(f"Error closing connection for {self.server_name}: {e}")